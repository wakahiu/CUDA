Work Division.

There is an equal number of streams as there are peices of data(second argument). 
Each stream transfers a piece of the data to and from the device.

On devices with 1 compute engine, all transfers to device memory are scheduled first 
followed by the kernel launches then transfers to host memory. This aims to everlap 
data tranfers to device with kernel exection of different streams. Finally when all the 
kernel finish, all data is transfered back.

On devices with 2 compute engines, we can overlap transfers to and from device memory. 
Therefore, each stream transfers a piece of the data followed by kernel launch then 
transfer of data back to host.

Timing Data
	Pieces	|	Time (MS)
	----------------------
	1		|	14.5
	2		|	11.3
	4		|	10.2
	8		|	9.51
***	16		|	9.06	<------ Best performance.
	32		|	9.22
	64		|	10.0
	128		|	16.8
	256		|	27.8
	512		|	50.63
	1024	|	97.87
	
From the timing data, an effective size of block occurs when we have 16 streams or 64*3KB of data


	
