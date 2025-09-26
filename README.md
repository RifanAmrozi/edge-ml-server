# Contribution Guide

Thanks for contributing 🎉! Please follow the workflow below to ensure smooth collaboration. 
Requirement model shoplifting: cvzone cv2 xgboost ultralytics pandas bumpy

---

## 📌 Git Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/rifanamrozi/edge-ml-server.git
   cd <repo>
   git remote add https://github.com/rifanamrozi/edge-ml-server.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b yourname/feat/my-new-feature
   ```

3. **Make Your Changes**  
   Write clean, tested, and well-documented code.

4. **Commit with Conventional Commits**
   - Example:
     ```bash
     git commit -m "feat(user): init user endpoint"
     ```
   - Format:  
     ```
     <type>(<scope>): <short description>
     ```
   - Common types:
     - `feat` → a new feature
     - `fix` → a bug fix
     - `docs` → documentation only
     - `style` → formatting, missing semicolons, etc.
     - `refactor` → code change that is not a fix or feature
     - `test` → adding or fixing tests
     - `chore` → maintenance tasks

5. **Rebase with Main Before PR**
   ```bash
   git stash
   git checkout main
   git pull
   git checkout yourname/feat/your-existing-pr
   git rebase origin/main
   ```

   If there are conflicts, resolve them and continue:
   ```bash
   git add .
   git rebase --continue
   ```

6. **Push Your Branch**
   ```bash
   git push origin yourname/feat/my-new-feature
   ```

7. **Open a Pull Request (PR)**
   - Go to your fork on GitHub.
   - Click **Compare & Pull Request**.
   - Fill in details about your changes.

---

## ✅ Example Commit Messages

- `feat(user): init user endpoint`
- `fix(auth): correct JWT expiration check`
- `docs(readme): update contribution guidelines`
- `refactor(api): simplify response handler`
- `test(user): add integration tests for login`

---

## 💡 Tips

- Keep PRs **small and focused**.
- Always write **meaningful commit messages**.
- Use **rebasing** (not merge) to keep history clean.
- Squash commits if necessary before merging.

---
