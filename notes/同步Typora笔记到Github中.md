# 同步Typora笔记到Github中

## 1.生成SSH密钥

```
ssh-keygen -t ed25519 -C "1019476315@qq.com"
```

## 2.**将密钥添加到 ssh-agent**：

```
# 启动ssh-agent
eval "$(ssh-agent -s)"

# 添加生成的私钥
ssh-add ~/.ssh/id_ed25519
```

## 3.**复制公钥内容**：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出的全部内容

![image-20251010172055361](../assests/同步Typora笔记到Github中/image-20251010172055361-1760088057193-7.png)

## 4.**添加到 GitHub 账户**：

- 登录 GitHub，点击右上角头像 → Settings

- 左侧菜单选择 SSH and GPG keys → New SSH key

- 在 "Title" 中输入一个标识（如 "我的 PC"）

- 在 "Key" 中粘贴刚才复制的公钥内容

- 点击 "Add SSH key" 保存

## 5.**验证配置**：

  ```bash
  ssh -T git@github.com
  ```

  如果成功，会显示类似 "Hi 你的用户名！You've successfully authenticated..." 的信息

  完成后，再次尝试克隆仓库就应该能成功了。

![image-20251010172113944](../assests/同步Typora笔记到Github中/image-20251010172113944.png)

