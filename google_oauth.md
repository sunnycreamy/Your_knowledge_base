# Google Drive API 配置指南

## 凭证获取步骤
1. 创建Google Cloud项目：
   - 访问 [Google Cloud Console](https://console.cloud.google.com/)
   - 点击右上角"创建项目"
   - 输入项目名称，点击"创建"

2. 启用Google Drive API：
   - 在左侧菜单选择"API和服务" > "库"
   - 搜索"Google Drive API"
   - 点击"启用"按钮

3. 创建OAuth凭证：
   - 在左侧菜单选择"API和服务" > "凭据"
   - 点击"创建凭据" > "OAuth 客户端ID"
   - 应用类型选择"桌面应用"
   - 输入应用名称
   - 下载JSON文件，重命名为`client_secret.json`
   - 将文件放置在项目根目录

4. 首次运行授权：
   - 运行程序时会自动打开浏览器
   - 选择你的Google账号
   - 点击"允许"授权访问
   - 授权成功后会自动生成`token.pickle`文件

## 注意事项
- `client_secret.json` 和 `token.pickle` 包含敏感信息，请勿上传到代码仓库
- 如需重新授权，删除 `token.pickle` 文件后重新运行程序
- 确保 Google 账号有足够的 Drive 权限

## 常见问题
1. 如果授权页面显示"未验证的应用"：
   - 选择"高级" > "继续访问"
   - 这是正常现象，因为应用处于开发状态

2. 如果遇到配额限制：
   - 访问 Google Cloud Console
   - 检查配额设置并根据需要申请提升

