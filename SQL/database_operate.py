# @Author        : Justin Lee
# @Time          : 2025-3-27

import sqlite3
import numpy as np

'''
    数据库操作
'''


# 创建数据库
def create_database(database_path: str=None):
    
    # 连接到指定路径的SQLite数据库（如果本来没有这个数据库文件，则会自动创建）
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    # 创建faces表，存储人脸图像、姓名和特征向量
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
    ''')
        
    connection.commit()
    connection.close()


# 把人脸特征向量和名字添加到数据库
def add_face_to_database(name, encoding, database_path: str=None):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
    connection.commit()
    connection.close()
    

# 检查要插入的姓名是否已存在数据库中
def check_name_unique(name: str, database_path: str):
    # 连接到数据库
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # 执行查询语句，检查 name 是否存在
    query = "SELECT COUNT(*) FROM faces WHERE name = ?"
    cursor.execute(query, (name,))
    result = cursor.fetchone()[0]

    # 关闭数据库连接
    conn.close()

    # 如果查询结果大于 0，说明 name 存在，返回 True；否则返回 False
    return result > 0


# 从SQLite数据库中加载已知人脸的特征向量和姓名
def load_known_faces(database_path: str=None):
    assert database_path is not None, 'Error: 请指定数据库文件路径！'
    
    known_face_encodings = []
    known_face_names = []

    # 连接指定的数据库
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    connection.close()
    
    for row in rows:
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float32)
        known_face_names.append(name)
        known_face_encodings.append(encoding)

    # 把所有的已知人脸embedding合成一个矩阵，方便后续计算相似度
    known_face_encodings = np.array(known_face_encodings)

    return known_face_encodings, known_face_names
