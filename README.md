# PPSTSL
The experiment focus on a novel privacy-preserving spatio-temporal graph data splitting learning method (PPSTSL) combining privacy protection and dynamic global nodes adjacency relationship. It designs the local-global spatio-temporal convolution operation(LGSTConv), each client obtains the global nodes adjacency relationship through the secure multi-party multiplication, then obtains the local spatial convolution features based on the global nodes adjacency relationship and the property of the block matrix, and finally obtains the global spatial convolution features through the secure aggregation method at the server, and inputs it into the global temporal convolution module to obtain the global spatio-temporal features.

# DataSet
The experimental dataset includes SZ_Taxi and Los_Loop, you can download it yourself on the Internet.

# Experimental environment
* pytorch_lightning == 1.6.0
* pandas
* torch == 1.10.1
* torch-cluser == 1.5.9
* torch-geometric == 2.1.0.post1
* torch-scatter == 2.0.9
* torch-sparse == 0.10.3
* numpy

# The project structure

│PPSTSTL   
├─models  
│    ├─_init_.py  
│    ├─gcn.py  
│    ├─gru.py  
│    ├─tgcn.py  
├─tasks  
│    ├─_init_.py  
│    ├─supervised.py  
└─utils  
│    ├─callbacks  
│    ├─data  
│    │   ├─_init_.py   
│    │   ├─function.py   
│    │   ├─spatiotemporal_csv_data.py   
│    ├─_init_.py  
│    ├─losses.py  
│    ├─metric.py  
│─main.py

# Acknowledgement  
This work was sponsored by the National Key Research and Development Program of China (No. 2018YFB0704400), Key Program of Science and Technology of Yunnan Province (No. 202002AB080001-2, 202102AB080019-3), Key Research Project of Zhejiang Laboratory (No.2021PE0AC02), Key Project of Shanghai Zhangjiang National Independent Innovation Demonstration Zone(No. ZJ2021-ZD-006). The authors gratefully appreciate the anonymous reviewers for their valuable comments.
