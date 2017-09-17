def delta_epsilon(a, b, e):
    return abs(a-b) < e


def generatePointSet(z, y, x,
                     minZRad,
                     maxZRad,
                     minYRad,
                     maxYRad,
                     minXRad,
                     maxXRad):
    center = (rand(0, z), rand(0, y), rand(0, x))
    toPopulate = []
    zRad = rand(minZRad, maxZRad)
    yRad = rand(minYRad, maxYRad)
    xRad = rand(minXRad, maxXRad)

    for cz in range(-zRad, zRad):
        for cy in range(-yRad, yRad):
            for cx in range(-xRad, xRad):
                curPoint = (center[0]+cz, center[1]+cy, center[2]+cx)
                #only populate valid points
                valid = True
                dimLimits = [z, y, x]
                for dim in range(3):
                    if curPoint[dim] < 0 or curPoint[dim] >= dimLimits[dim]:
                        valid = False
                if valid:
                    toPopulate.append(curPoint)
    return set(toPopulate)


def generateTestVolume(z, y, x,
                       minCluster,
                       maxCluster,
                       minZRad,
                       maxZRad,
                       minYRad,
                       maxYRad,
                       minXRad,
                       maxXRad):
    #create a test volume
    volume = np.zeros((z, y, x))
    myPointSet = set()
    for _ in range(rand(minCluster, maxCluster)):
        potentialPointSet = generatePointSet(z, y, x,
                                             minZRad,
                                             maxZRad,
                                             minYRad,
                                             maxYRad,
                                             minXRad,
                                             maxXRad)
        #be sure there is no overlap
        while len(myPointSet.intersection(potentialPointSet)) > 0:
            potentialPointSet = generatePointSet(z, y, x,
                                                 minZRad,
                                                 maxZRad,
                                                 minYRad,
                                                 maxYRad,
                                                 minXRad,
                                                 maxXRad)

        for elem in potentialPointSet:
            myPointSet.add(elem)

    #populate the true volume
    for elem in myPointSet:
        volume[elem[0], elem[1], elem[2]] = 1

    return volume, list(myPointSet)


def visualize_volume(volume):
    points = np.nonzero(volume)
    zs = points[0]
    ys = points[1]
    xs = points[2]

    fig = plt.figure()

    ax1=fig.add_subplot(2,2,1, projection='3d')
    ax2=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(2,2,3)
    ax4=fig.add_subplot(2,2,4)

    ax1.set_title('3D View')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.scatter(xs, ys, zs)

    ax2.set_title('xy View')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.scatter(xs, ys)

    ax3.set_title('xz View')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.scatter(xs, zs)

    ax4.set_title('yz View')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')
    ax4.scatter(ys, zs)


    plt.tight_layout(pad=1.5)
    plt.show()


if __name__ == '__main__':
    print(delta_epsilon(1., 1.1, .2) == True)
    print(delta_epsilon(1., 1.1, .01) == False)