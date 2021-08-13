__constant sampler_t sampler_const =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST;

__kernel void blur(
	__constant float * krnl, 
	__private int kernelSize,
	read_only image2d_t source,
	write_only image2d_t dest)
{
    // Get current pixel
    const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));

    float currPix_x = 0.0f;
    float currPix_y = 0.0f;
    float currPix_z = 0.0f;
    int2 currentKernelPos;

    for(int i = -kernelSize; i < kernelSize+1; i++) {
	for(int j = -kernelSize; j < kernelSize+1; j++) {
	    currentKernelPos = (int2)(i, j);
	    
	    // Get kernel position from the array
	    int x = i + kernelSize;
	    int yW = (j + kernelSize) * (kernelSize*2+1);

	    // Calculate each color for current pixel
	    currPix_x += krnl[x + yW] 
		* read_imagef(source, sampler_const, pixel_id + (int2)(i, j)).x;

	    currPix_y += krnl[x + yW] 
		* read_imagef(source, sampler_const, pixel_id + (int2)(i, j)).y;

	    currPix_z += krnl[x + yW] 
		* read_imagef(source, sampler_const, pixel_id + (int2)(i, j)).z;
	}
    }

    // Save the output buffer with calculated color channels
    write_imagef(dest, pixel_id, (float4)(currPix_x, currPix_y, currPix_z, 1.0));
}
