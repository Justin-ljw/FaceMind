import sqlite3
import numpy as np

'''
    从SQLite数据库中加载已知人脸的特征向量和姓名
'''

def load_known_faces(database_path: str=None):
    assert database_path is not None, 'Error: 请指定数据库文件路径！'
    
    known_face_encodings = []
    known_face_names = []

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

    return known_face_encodings, known_face_names