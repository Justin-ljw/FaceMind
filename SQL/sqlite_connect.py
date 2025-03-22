import sqlite3
import os

'''
    获取SQLite数据库连接，避免重复创建连接，提高性能
'''

def get_connection(database_path: str=None) -> sqlite3.Connection:
    
    assert database_path is not None, 'Error: 请指定数据库文件路径！'
    
    # 连接到指定路径的SQLite数据库（如果本来没有这个数据库文件，则会自动创建）
    connection = sqlite3.connect(database_path)
    
    # 检查数据库文件是否存在，如果不存在则初始化数据库
    if not os.path.exists(database_path):
        cursor = connection.cursor()
        
        # 创建faces表，存储人脸图像、姓名和特征向量
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
        ''')
        
        connection.commit()
    
    # 返回数据库连接对象
    return connection
