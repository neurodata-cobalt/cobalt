# To run N4ITK bias correction
1. ssh onto cortex with local 8888 port forwarding
2. Start the docker by running `docker run -v /media/vikram/braindrive3/:/run/data/ -p 8888:8888 neurodata/ndreg` 
  * Note that it has to be ran with this specific volume
  * If the docker image is not found (you can check this by typing `docker images` and look for neurodata/ndreg) then pull it by running `docker pull neurodata/ndreg` and trying the step again.
  * Check if the docker is already running by running `docker ps`. If you see the container `neurodata/ndreg` then kill it by running `docker kill <container-id>` and run the command again. Container id is found in the first column when you run `docker ps`.
3. The terminal should give you a link to paste into your browser and access the jupyter environment. 
4. Start the notebook by navigating to `data/N4 Correction.ipynb`
5. Run the first 3 blocks (up to **Running N4ITK**). Let it run for about 30 seconds. You should see output like 
  * **On location raw_data/LOC005
    Done with image VW0_LOC005D_CM0_CHN01_PLN0107 in 0.194036006927 seconds**
6. Pres `ii` to interrupt (or interupt?) then run the last cell to visualize some of the slices before and after correction. The point is to just show N4ITK is running successfully on a few images. Running N4ITK on everything will take days. 

# To run Terastitcher
1. ssh onto cortex with local 8888 port forwarding
2. Run `/home/jyim6/.local/bin/teraconverter --sfmt="TIFF (unstitched, 3D)" -s=../xml_merging.xml --dfmt="TIFF (series, 2D)"  -d=../s3617_demo_tiles/` 
  * You might run into permission errors. In that case please contact me :). 
3. You should see a notification that says teraconverter has started with a progress bar. It'll stay at 0% for a while because it's 
computing how much data and how long it estimates it will take. You can wait to see it start to stitch or just believe me that this
is what I did. 

# BOSS visualization
To visualize s3617 with N4ITK ran on the image after it was stitched go to [ndvis](https://viz.boss.neurodata.io/#!{'layers':{'s3617_res4':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/bias_corrections/s3617_corrected_res4/s3617_res4?window=0,10000'_'max':0.14}_'s3617_scale_025_fwhm_0100':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/bias_corrections/s3617_corrected_res4/s3617_scale_025_fwhm_0100?window=0,10000'_'opacity':0.5_'color':1}}_'navigation':{'pose':{'position':{'voxelSize':[9360_9360_5000]_'voxelCoordinates':[461_655.5_649]}}_'zoomFactor':20000}})

To visualize s3617 with entropy ran on it then stitched, go to [ndvis](https://auth.boss.neurodata.io/auth/realms/BOSS/protocol/openid-connect/auth?client_id=endpoint&redirect_uri=https%3A%2F%2Fviz.boss.neurodata.io%2F%3Fredirect_fragment%3D!%257B%27layers%27%253A%257B%27Ch0_daniel_bias_corrected%27%253A%257B%27type%27%253A%27image%27_%27source%27%253A%27boss%253A%252F%252Fhttps%253A%252F%252Fapi.boss.neurodata.io%252Fbias_corrections%252Fs3617_daniel_bias_corrected%252FCh0_daniel_bias_corrected%253Fwindow%253D0%252C250%27%257D%257D_%27navigation%27%253A%257B%27pose%27%253A%257B%27position%27%253A%257B%27voxelSize%27%253A%255B585_585_5000%255D_%27voxelCoordinates%27%253A%255B16004.8955078125_12830.9375_619%255D%257D%257D_%27zoomFactor%27%253A18524.190766443455%257D%257D&state=b16d4b6a-239e-4b3e-8ddf-ebc42d3bc1ee&nonce=fecc76b8-995a-4fc9-8ca6-0f687cdc2597&response_mode=query&response_type=code)

* If you only see black, you may have to decrease the brightness range to see anything. Right click on the channel and decrease the max slider.
* Probably have to zoom out.
* Change window range to [0,5000]
* If any other problems contact me.
