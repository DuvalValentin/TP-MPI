## docker.openmpi

With the code in this repository, you can build a Docker container that provides 
the OpenMPI runtime and tools along with various supporting libaries, 
including the MPI4Py Python bindings. The container also runs an OpenSSH server
so that multiple containers can be linked together and used via `mpirun`.


## Start an MPI Container Cluster

While containers can in principle be started manually via `docker run`, we suggest that your use 
[Docker Compose](https://docs.docker.com/compose/), a simple command-line tool 
to define and run multi-container applications. We provde a sample `docker-compose.yml`
file in the repository:

```
mpi_head:
  image: openmpi
  ports: 
   - "22"
  links: 
   - mpi_worker

mpi_node: 
  image: openmpi

```

The file defines an `mpi_head` and an `mpi_node`. Both containers run the same `openmpi` image. 
The only difference is, that the `mpi_head` container exposes its SHH server to 
the host system, so you can log into it to start your MPI applications.


The following command will start one `mpi_head` container and three `mpi_node` containers: 

```
$> docker-compose scale mpi_head=1 mpi_node=3
```
Once all containers are running, figure out the host port on which Docker exposes the  SSH server of the  `mpi_head` container: 

```
$> 
```

Now you know the port, you can login to the `mpi_head` conatiner. The username is `mpirun`:

> TODO: Password

 ```
 $> ssh -p 23227 mpirun@localhost
 ```
