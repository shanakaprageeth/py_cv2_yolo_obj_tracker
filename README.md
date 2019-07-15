# Object counter and tracking using yolo

This script allow object detection and tracking through trained yolo framework using opencv.



### Prerequisites

python 3.5.2
opencv for python
numpy

### Installing

Please use following commands download and setup yolo dataset
``` 
./set_env.sh
``` 

## Running the tests

Run the test using
``` 
python object_couter.py
``` 
Please set following variables according to your requirnments
``` 
# save_csv
SAVE_CSV = True
# track objects
TRACK = True
# save output video
VID_OUT = True
``` 
Other options 
``` 
python object_couter.py -h for additional options
``` 

## License
[MIT](https://choosealicense.com/licenses/mit/)