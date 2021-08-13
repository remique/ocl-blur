extern crate image;
extern crate ocl;

use ocl::enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType};
use ocl::{builders::DeviceSpecifier, Buffer, Image, ProQue};
use std::fs;
use std::path::Path;

fn create_matrix(radius: i32) -> Vec<f32> {
    let width = ((radius * 2) + 1) as usize;
    let mut mtrx = vec![vec![0.0f32; width]; width];
    let sigma: f32 = (radius as f32 / 2.0).max(1.0);
    let e = std::f32::consts::E;
    let pi = std::f32::consts::PI;
    let mut sum = 0.0;

    for x in -radius..radius {
        for y in -radius..radius {
            let exp_nom: f32 = -((x * x) + (y * y)) as f32;
            let exp_den: f32 = 2.0 * sigma * sigma;

            let e_expr: f32 = e.powf(exp_nom / exp_den);
            let e_val: f32 = e_expr / (2.0 * pi * sigma * sigma);

            let i = (x + radius as i32) as usize;
            let j = (y + radius as i32) as usize;

            mtrx[i][j] = e_val;
            sum += mtrx[i][j];
        }
    }

    // Normalize the kernel so that sum = 1
    for x in 0..width {
        for y in 0..width {
            mtrx[x][y] /= sum;
        }
    }

    // Now convert it to the Vec<f32>
    let flatten_mtrx: Vec<f32> = mtrx
        .iter()
        .flat_map(|nested| nested.iter())
        .cloned()
        .collect();

    flatten_mtrx
}

fn load_image(path: &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let img = image::open(&Path::new(path)).unwrap().to_rgba();

    img
}

fn main() -> ocl::Result<()> {
    // Load source file and make it a string
    let src_file = fs::read("src/kernel.cl")?;
    let src_string = String::from_utf8(src_file).unwrap();

    // // Load source image (host)
    let source_img = load_image("test.jpg");
    let dims = source_img.dimensions();

    let pro_que = ProQue::builder()
        .src(src_string)
        .device(DeviceSpecifier::TypeFlags(ocl::flags::DEVICE_TYPE_GPU))
        .dims(&dims)
        .build()
        .unwrap();

    // // This will be the output image (host)
    let mut result: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::new(dims.0, dims.1);

    // // Source image buffer (GPU)
    let cl_source = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .queue(pro_que.queue().clone())
        .copy_host_slice(&source_img)
        .build()
        .unwrap();

    // // Destination image buffer (GPU)
    let cl_destination = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_WRITE_ONLY
                | ocl::flags::MEM_HOST_READ_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .queue(pro_que.queue().clone())
        .copy_host_slice(&result)
        .build()
        .unwrap();

    let blur_kernel_radius = 10;
    let vec_blur_kernel = create_matrix(blur_kernel_radius);
    let blur_kernel_length = vec_blur_kernel.len();
    let blur_kernel_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(blur_kernel_length)
        .copy_host_slice(&vec_blur_kernel)
        .build()?;

    // Set up the kernel
    let kernel = pro_que
        .kernel_builder("blur")
        .arg(&blur_kernel_buffer)
        .arg(blur_kernel_radius)
        .arg(&cl_source)
        .arg(&cl_destination)
        .build()?;

    // Enqueue the kernel
    unsafe {
        kernel.enq()?;
    }

    // // Copy output buffer to the result image on the host
    cl_destination.read(&mut result).enq()?;

    // // Save the image
    result.save(&Path::new("result.png")).unwrap();

    Ok(())
}
