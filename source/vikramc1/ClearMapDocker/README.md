# Running the Dockerfile

In order to test ClearMap with your image volume, run the following to build the Dockerfile:

`docker build -t clearmap .`

Now in order to give the Docker image access to your image volume run the following replacing `path/to/your/image` with the path to the folder containing your image:

`docker run --rm -v path/to/your/image:/run/data -p 8888:8888 clearmap`

Then, go to the link printed out in the terminal in your web browser. Navigate into the `data` folder and click on the Jupyter notebook.
