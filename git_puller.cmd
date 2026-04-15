echo "switching to branch juan"
git switch juan
timeout /t 5
git pull
timeout /t 1
git merge origin/main
timeout /t 2


echo "switching to branch romeo"
git switch romeo
timeout /t 5
git pull
timeout /t 1
git merge origin/main
timeout /t 2

echo "switching to branch alberto"
git switch alberto
timeout /t 5
git pull
timeout /t 1
git merge origin/main
timeout /t 2

echo "switching to branch gang"
git switch gang
timeout /t 5
git pull
timeout /t 1
git merge origin/main
timeout /t 2

echo "end of all branches"
git switch main
timeout /t 10