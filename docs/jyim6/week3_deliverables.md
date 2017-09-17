# Week 3 deliverables

### Getting data
* Along with team, got experienced with BOSS and wrote a script to retrieve a tif-converted brain image.
### Vaa3d toubleshooting
* At first tried my best to [build Vaa3D from scratch](https://github.com/Vaa3D/Vaa3D_Wiki/wiki/BuildVaa3d.wiki)
* After many hours of crashes and Qt incompatibility issues, switched to using the binary relases. 
** Had to trouble shoot here because all the stable releases don't work with the latest Mac OS. 
For anyone interested in using Vaa3D and uses a mac, follow the forum thread [here](https://www.nitrc.org/forum/forum.php?thread_id=7925&forum_id=1553)
where the author links the latest unreleased version of Vaa3D that fixes the current bugs.

### Messing around with Vaa3D
* Played around with loading image data and using their cell count plugins. Results:
![](https://user-images.githubusercontent.com/8682187/30517504-a5a70d6a-9b2f-11e7-9d90-16dbe5f5b5d5.png)
The plug in works by first select some expemplar neurons by clicking on the center of 5-10 cells that you know are neurons. 
The algorithm then uses these markers you selected to find other cells in the volume that look similar to it. 
As you see while a lot of cells are marked, there are also a lot of cells that are not. 

* Also played with their cell tracing plug in:
![](https://user-images.githubusercontent.com/8682187/30517497-967cc19a-9b2f-11e7-8b76-1401f9d8ca10.png)
![](https://user-images.githubusercontent.com/8682187/30517501-a28a9a20-9b2f-11e7-8c7e-3a55851002c7.png)
The tracing software seems useful and well done. It runs reasonably fast. However I don't have a way of validating how well it works 
just based on looking. The results of cell tracing seem to vary depending on how many cells are detected.
The results of this seems like good background for tractography and cell-counting algorithms.
