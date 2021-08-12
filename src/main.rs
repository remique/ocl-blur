extern crate image;
extern crate ocl;

use ocl::enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType};
use ocl::{builders::DeviceSpecifier, Image, ProQue};
use std::path::Path;

fn create_matrix(radius: i32) {
    let width = (radius * 2) as usize;
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

    for x in 0..width {
        for y in 0..width {
            mtrx[x][y] /= sum;
        }
    }

    println!("{:?}", mtrx);

    // Now convert it to the float* ( so [f32] slice )
}

fn load_image(path: &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let load = image::open(&Path::new(path)).unwrap();
    let img = load.to_rgba();

    img
}

fn main() -> ocl::Result<()> {
    let src = r#"
    __constant sampler_t sampler_const =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_NEAREST;

    __kernel void copy_img(read_only image2d_t source, write_only image2d_t dest) {
        // buffer[get_global_id(0)] += scalar;
        //
        // Get current pixel
        const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));

        // Read the pixel into float4 value
        const float4 rgba = read_imagef(source, sampler_const, pixel_id);

        // Copy original contents into the output
        write_imagef(dest, pixel_id, (float4)(rgba.x, rgba.y, rgba.z, 1.0));
    }
    "#;

    // Load source image (host)
    let source_img = load_image("test.jpg");
    let dims = source_img.dimensions();

    let pro_que = ProQue::builder()
        .src(src)
        .device(DeviceSpecifier::TypeFlags(ocl::flags::DEVICE_TYPE_GPU))
        .dims(&dims) // 64 dec
        .build()?;

    let buffer = pro_que.create_buffer::<f32>()?;

    // This will be the output image (host)
    let mut result: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::new(dims.0, dims.1);

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

    let kernel = pro_que
        .kernel_builder("copy_img")
        .arg(&cl_source)
        .arg(&cl_destination)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    cl_destination.read(&mut result).enq()?;

    result.save(&Path::new("result.png")).unwrap();

    Ok(())

    // create_matrix(2);

    // let nested_array = [
    //     [17, 16, 15, 14, 13],
    //     [18, 5, 4, 3, 12],
    //     [19, 6, 1, 2, 11],
    //     [20, 7, 8, 9, 10],
    //     [21, 22, 23, 24, 25],
    // ];
    // let flatten_array: Vec<i32> = nested_array
    //     .iter()
    //     .flat_map(|array| array.iter())
    //     .cloned()
    //     .collect();

    // let mtrx = [
    //     [1, 2, 3],
    //     [4, 5, 6],
    //     [7, 8, 9],
    // ];

    // println!("{:?}", flatten_array);
}
