# Running the Dockerfile

In order to test ClearMap with your image volume, run the following to build the Dockerfile. If you have already built the Dockerfile, you don't need to build it again (i.e. run this command only the first time):

`docker build -t clearmap .`

Now in order to give the Docker image access to your image volume run the following replacing `path/to/your/image` with the path to the folder containing your image:

`docker run --rm -v path/to/folder/containing/image/:/run/data -p 8888:8888 clearmap`

Then, go to the link printed out in the terminal in your web browser. Navigate into the `data` folder and click on the Jupyter notebook.
