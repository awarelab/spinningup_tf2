echo "\n\tChecking code with pylint..."
echo "\t============================\n"
find . -type f -name "*.py" | xargs pylint

echo "\n\tChecking code with isort..."
echo "\t===========================\n"
isort -rc . --diff --check-only
