import importlib.util





###################### 不同文件夹下相同的py名里相同名的class导入为不同的类名
############# folder1.lwa.agent        folder2.lwa.agent    folder3.lwa.agent





# 假设你的类所在的模块路径是 '/path/to/module/module_name.py'
folders = ['./folder1/lwa.py','./folder2/lwa.py','./folder3/lwa.py']
classname = {}
for i,folder in enumerate(folders):

    module_name = folder.replace('/', '.').replace('.py', '')


    # 创建模块规范
    spec = importlib.util.spec_from_file_location(module_name, folder)

    # 创建模块
    module = importlib.util.module_from_spec(spec)

    # 执行模块
    spec.loader.exec_module(module)

    # 现在你可以从模块中获取类
    classname[i] = module.Agent

agent1 = classname[0]("Agent1")
agent2 = classname[1]("Agent2")
agent3 = classname[2]("Agent3")

print(agent1.greet())
print(agent2.greet())
print(agent3.greet())