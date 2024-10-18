# collabml_client

This is the code of the client side of HAPI, a processing system for transfer learning that spans the compute and the object storage tiers.
HAPI is presented in the paper "Accelerating Transfer Learning with Near-Data Computation on Cloud Object Stores", co-authored by Diana Petrescu, Arsany Guirguis, Do Le Quoc, Javier Picorel, Rachid Guerraoui and Florin Dinu.
HAPI is built on top of OpenStack swift, the open-source object storage, and PyTorch for ML computations.

## Setup
HAPI is composed of two components: `server` and `client`. In our experiments, we test with 2 machines, each host a separate component.
Please use the [server guide](server_setup_guide.md) to set up the server and the [client guide](client_setup_guide.md) to set up the client.

## Running the server
After a successful setup of the server, run `rebuildswift` to start it. You can run `swift-init all stop` to stop it.

To run the HAPI server outside of the proxy server (refer to the paper for more information), run on the server machine:
```
python3 $HOME/swift/swift/proxy/mllib/server.py
```
**Important**: make sure to update the server IP in [this line](https://github.com/aguirguis/swift/blob/ml-swift/swift/proxy/mllib/server.py#L325).

## Sample runs
On the client side, you can run the following examples:
```
#split experiment
cd $HOME/collabml_client/application_layer; python3 main.py --dataset imagenet --model myalexnet --num_epochs 1 --batch_size 8000 --freeze --freeze_idx 17 --use_intermediate; cd -
#vanilla experiment
cd $HOME/collabml_client/application_layer; python3 main.py --dataset imagenet --model alexnet --num_epochs 1 --batch_size 8000 --freeze --freeze_idx 17; cd -
#swift-only baseline (called ALL_IN_COS in our paper)
cd $HOME/collabml_client; python3 mlswift_playground.py --dataset imagenet --model alexnet --num_epochs 1 --batch_size 1000 --task training --freeze_idx 17; cd -
```

## Paper experiments
The scripts to reproduce the results and the plots of our paper can be found in [the experiments directory](application_layer/experiments). Please see `run_exp.py` for all scripts that we use to run the paper experiments and `main.py` for the plotting scripts.

## References
CLI documentation can be found [here](https://docs.openstack.org/python-swiftclient/latest/cli/index.html) and the client Service API documentation is [here](https://docs.openstack.org/python-swiftclient/latest/service-api.html).

## Questions
If you have any question, do not hesitate to contact me on arsany.guirguis@epfl.ch
