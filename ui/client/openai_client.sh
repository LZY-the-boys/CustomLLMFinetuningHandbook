curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "lu-vae/qwen-sharegpt4-merged",
"messages": [
{"role": "system", "content": ""},
{"role": "user", "content": "Write a Perl script that processes a log file and counts the occurrences of different HTTP status codes. The script should accept the log file path as a command-line argument and print the results to the console in descending order of frequency."}
]
}'| echo -e "$(cat)\n" >> tmp.json