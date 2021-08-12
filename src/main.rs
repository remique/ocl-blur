extern crate ocl;
use ocl::{builders::DeviceSpecifier, ProQue};

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

fn main() -> ocl::Result<()> {
    // fn main() {
    let src = r#"
    __kernel void add(__global float* buffer, float scalar) {
        buffer[get_global_id(0)] += scalar;
    }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .device(DeviceSpecifier::TypeFlags(ocl::flags::DEVICE_TYPE_GPU))
        .dims(1 << 6) // 64 dec
        .build()?;

    let buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("add")
        .arg(&buffer) // float* buffer
        .arg(10.0f32) // float scalar
        .build()?;

    println!("Context: {:?}", pro_que.context().devices()[0].name());
    println!("Device: {:?}", pro_que.device().name());

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    Ok(())

    //     println!("The value at index [{}] is now '{}'!", 0, vec[0]);
    //     println!("The value at index [{}] is now '{}'!", 1, vec[1]);
    //     println!("The value at index [{}] is now '{}'!", 2, vec[2]);

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
