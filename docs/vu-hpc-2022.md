# VU HPC Course 2022

Links:

[1] VU HPC Course 2022 https://hpc.labs.vu.nl/schedule-2022/

[2] SURF User Knowledge Base https://servicedesk.surf.nl/wiki/display/WIKI/SURF+User+Knowledge+Base

## SURFLisa

### Apply for account

For a regular cpu Lisa account a mail to helpdesk@surfsara.nl is sufficient (see https://bit.ly/SURF-Lisa )

To apply for the Lisa GPU island use the VU/UvA form at https://servicedesk.surf.nl/

In Service Desk, we can apply for different resources of SURF, such as LISA, Snellius, Research Cloud etc.

### Login

cpu node: `username@lisa.surfsara.nl`

gpu node: `username@login-gpu.lisa.surfsara.nl`

### Transfer data

```shell
# copy from local PC to HPC system
scp sourcefile <username>@lisa.surfsara.nl:destinationdir  
scp -r sourcedir <username>@lisa.surfsara.nl:destinationdir

# copy from HPC system to local PC
scp -r <username>@lisa.surfsara.nl:sourcefile ~
```

### Writing a job



## SURF Cloud

