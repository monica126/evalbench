export PYTHONPATH=./evalbench:./evalbench/evalproto
python3 evalbench/client/eval_client.py --experiment="evalbench/$EVAL_CONFIG"
