use crate::sys;

use candle::{
    CudaStorage, CustomOp1, DType, Layout, Result, Shape, Storage, Tensor,
    backend::BackendStorage,
    cuda::{
        CudaDType, WrapErr as _,
        cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
    },
};

pub fn nvte_fused_rope(
    input: &Tensor,
    cu_seqlens: &Tensor, //[i32]
    freqs: &Tensor,
    start_positions: &Tensor, //[i32]
    interleaved: bool,
) -> Result<Tensor> {
    println!("fused rope input = {:?}", input);
    println!("cu_seqlens = {}", cu_seqlens);
    println!("freqs = {:?}", freqs);
    println!("start_positions = {}", start_positions);
    println!("interleaved = {:?}", interleaved);

    let rope = NVTEFusedRope::new(
        cu_seqlens.clone(),
        freqs.clone(),
        start_positions.clone(),
        interleaved,
    );

    input.apply_op1(rope)
}

fn nvte_dtype(dtype: DType) -> sys::NVTEDType {
    match dtype {
        DType::BF16 => sys::NVTEDType_kNVTEBFloat16,
        DType::U32 => sys::NVTEDType_kNVTEInt32,
        DType::I64 => sys::NVTEDType_kNVTEInt64,
        DType::F32 => sys::NVTEDType_kNVTEFloat32,
        _ => panic!("unsupported nvte dtype: {:?}", dtype),
    }
}

//TODO impl Drop for NVTETensor
pub struct NVTETensor {
    pub ptr: sys::NVTETensor,
}

impl NVTETensor {
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let (storage, layout) = tensor.storage_and_layout();

        match &*storage {
            Storage::Cuda(cuda_storage) => match tensor.dtype() {
                DType::U8 => Self::from_storage::<u8>(cuda_storage, &layout),
                DType::U32 => Self::from_storage::<u32>(cuda_storage, &layout),
                DType::I64 => Self::from_storage::<i64>(cuda_storage, &layout),
                DType::BF16 => Self::from_storage::<half::bf16>(cuda_storage, &layout),
                DType::F16 => Self::from_storage::<half::f16>(cuda_storage, &layout),
                DType::F32 => Self::from_storage::<f32>(cuda_storage, &layout),
                DType::F64 => Self::from_storage::<f64>(cuda_storage, &layout),
            },
            _ => panic!("nvte only supports cuda"),
        }
    }

    pub fn from_slice<T: CudaDType + DeviceRepr>(
        slice: &CudaSlice<T>,
        layout: &Layout,
        dtype: DType,
        stream: &CudaStream,
    ) -> Result<Self> {
        let (ptr, _sync) = slice.device_ptr(stream);

        let mut shape_data = [0; 15];
        for (i, d) in layout.shape().dims().iter().enumerate() {
            shape_data[i] = *d as usize;
        }

        let shape = sys::NVTEShape {
            data: shape_data,
            ndim: layout.shape().dims().len(),
        };

        // let shape =
        let basic_tensor = sys::NVTEBasicTensor {
            data_ptr: ptr as *mut _,
            dtype: nvte_dtype(dtype),
            shape,
        };

        let nvte_tensor_ptr = unsafe {
            // note "delayed" scaling = unquantized
            let mut t = sys::nvte_create_tensor(sys::NVTEScalingMode_NVTE_DELAYED_TENSOR_SCALING);
            sys::nvte_set_tensor_param(
                &mut t,
                sys::NVTETensorParam_kNVTERowwiseData,
                &basic_tensor,
            );
            t
        };

        Ok(Self {
            ptr: nvte_tensor_ptr,
        })
    }

    pub fn from_storage<T: CudaDType + DeviceRepr>(
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<Self> {
        let stream = storage.device().cuda_stream();
        let dtype = storage.dtype();
        Self::from_slice(storage.as_cuda_slice::<T>()?, layout, dtype, &stream)
    }
}

pub struct NVTEFusedRope {
    pub cu_seqlens: Tensor,
    pub freqs: Tensor,
    pub start_positions: Tensor,
    pub interleaved: bool,
}

impl NVTEFusedRope {
    pub fn new(
        cu_seqlens: Tensor,
        freqs: Tensor,
        start_positions: Tensor,
        interleaved: bool,
    ) -> Self {
        Self {
            cu_seqlens,
            freqs,
            start_positions,
            interleaved,
        }
    }

    fn cuda_fwd_t<T: CudaDType + DeviceRepr>(
        &self,
        input: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let stream = input.device().cuda_stream();
        let cu_seqlens_nvte_tensor = NVTETensor::from_tensor(&self.cu_seqlens)?;
        let start_positions_nvte_tensor = NVTETensor::from_tensor(&self.start_positions)?;
        let freqs_nvte_tensor = NVTETensor::from_tensor(&self.freqs)?;
        let input_nvte_tensor = NVTETensor::from_storage::<T>(input, layout)?;
        let output = unsafe { stream.alloc::<T>(layout.shape().elem_count()).w()? };
        let output_nvte_tensor =
            NVTETensor::from_slice::<T>(&output, layout, input.dtype(), &stream)?;

        let qkv_format = sys::NVTE_QKV_Format_NVTE_THD;

        let (full_t, h, d) = layout.shape().dims3()?;
        let b_sz = self.start_positions.shape().dims1()?;
        let t = full_t / b_sz;

        let d2 = if self.interleaved { d } else { d / 2 };
        let cp_size = 1;
        let cp_rank = 0;

        let stride_d = 1;
        let stride_h = d;
        let stride_t = h * d;
        let stride_b = 0;

        dbg!(
            t, b_sz, h, d, d2, cp_size, cp_rank, stride_t, stride_b, stride_h, stride_d
        );

        unsafe {
            sys::nvte_fused_rope_forward(
                input_nvte_tensor.ptr as *mut _,
                cu_seqlens_nvte_tensor.ptr as *mut _,
                freqs_nvte_tensor.ptr as *mut _,
                start_positions_nvte_tensor.ptr as *mut _,
                output_nvte_tensor.ptr as *mut _,
                qkv_format,
                self.interleaved,
                cp_size as i32,
                cp_rank as i32,
                t as i32,
                b_sz as i32,
                h as i32,
                d as i32,
                d2 as i32,
                stride_t as i32, // stride_s
                stride_b as i32, // stride_b
                stride_h as i32, // stride_h
                stride_d,        // stride_d
                stream.cu_stream() as sys::cudaStream_t,
            );
        };

        let out = CudaStorage::wrap_cuda_slice(output, input.device().clone());
        Ok((out, layout.shape().clone()))
    }
}

impl CustomOp1 for NVTEFusedRope {
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        match storage.dtype() {
            candle::DType::BF16 => self.cuda_fwd_t::<half::bf16>(storage, layout),
            candle::DType::F32 => self.cuda_fwd_t::<f32>(storage, layout),
            _ => panic!("nvte_fused_rope only supports bf16"),
        }
    }

    fn name(&self) -> &'static str {
        "nvte_fused_rope"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        panic!("nvte_fused_rope is cuda only");
    }
}
