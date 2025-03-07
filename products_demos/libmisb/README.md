# LibMISB

LibMISB is a library that supports the encoding and decoding of MISB standards using the Key-Length-Value (KLV) protocol. KLV is a data encoding structure for transmission or storage. This library will help you to manage the encoding and decoding procedures in a simple way following MISB standards. 

You can find more information about LibMISB in our developer's wiki:

[LibMISB developer's wiki](https://developer.ridgerun.com/wiki/index.php/LibMISB)

You can purchase LibMISB product at:

[RidgeRun's store](https://shop.ridgerun.com/products/libmisb)

## Producer consumer sample
The purpose of this sample is to show how to encode a message using LibMISB, the producer will update the timestamp every second and
send it to the consumer which is listening through a pipe

### Compilation instructions

```
meson setup build --prefix=/usr
ninja -C build
sudo ninja -C build install
```

### Execution instructions

```
# Start the consumer
libmisb_app
```

```
# Start the producer in another terminal
libmisb_app -p
```

The program will start showing the encoded and decoded results like the following

```bash
./build/libmisb_app -p
Producer mode
Timestamp: Oct. 18, 2024. 08:20:33.770
Encoded data: 
6 e 2b 34 2 b 1 1 e 1 3 1 1 0 0 0 2c 2 8 0 6 24 c1 1 90 e 10 3 9 4d 49 53 53
49 4f 4e 30 31 4 6 41 46 2d 31 30 32 5 2 71 c2 f 2 c2 21 41 1 11 1 2 45 88 

Timestamp: Oct. 18, 2024. 08:20:34.770
Encoded data: 
6 e 2b 34 2 b 1 1 e 1 3 1 1 0 0 0 2c 2 8 0 6 24 c1 1 9f 50 50 3 9 4d 49 53 53
49 4f 4e 30 31 4 6 41 46 2d 31 30 32 5 2 71 c2 f 2 c2 21 41 1 11 1 2 94 ca 

Timestamp: Oct. 18, 2024. 08:20:35.771
Encoded data: 
6 e 2b 34 2 b 1 1 e 1 3 1 1 0 0 0 2c 2 8 0 6 24 c1 1 ae 96 78 3 9 4d 49 53 53
49 4f 4e 30 31 4 6 41 46 2d 31 30 32 5 2 71 c2 f 2 c2 21 41 1 11 1 2 cc 10 

...

```

```bash
./build/libmisb_app 
Consumer mode
...INFO	Formatted data generated 

Decoded metadata:
{"key": "060E2B34020B01010E01030101000000", "items": [{"tag": "2", "value": "Oct. 18, 2024. 08:20:33.770"},
{"tag": "3", "value": "MISSION01"}, {"tag": "4", "value": "AF-102"}, {"tag": "5", "value": "159.974365"},
{"tag": "15", "value": "14190.719463"}, {"tag": "65", "value": "17"}]}
..INFO	Formatted data generated 

Decoded metadata:
{"key": "060E2B34020B01010E01030101000000", "items": [{"tag": "2", "value": "Oct. 18, 2024. 08:20:34.770"},
{"tag": "3", "value": "MISSION01"}, {"tag": "4", "value": "AF-102"}, {"tag": "5", "value": "159.974365"},
{"tag": "15", "value": "14190.719463"}, {"tag": "65", "value": "17"}]}
..INFO	Formatted data generated 

Decoded metadata:
{"key": "060E2B34020B01010E01030101000000", "items": [{"tag": "2", "value": "Oct. 18, 2024. 08:20:35.771"},
{"tag": "3", "value": "MISSION01"}, {"tag": "4", "value": "AF-102"}, {"tag": "5", "value": "159.974365"},
{"tag": "15", "value": "14190.719463"}, {"tag": "65", "value": "17"}]}
..INFO	Formatted data generated 
...
```
