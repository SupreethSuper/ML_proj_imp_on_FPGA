git switch alberto
git pull
git add .
set /p msg = Enter the commit message : 
git commit -m "%msg"
git push --force