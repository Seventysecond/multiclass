import cx_Oracle
import time

def test_network_speed(connection_string, sql):
    try:
        connection = cx_Oracle.connect(connection_string)
        cursor = connection.cursor()

        start_time = time.time()
        cursor.execute(sql)
        data = cursor.fetchall()
        end_time = time.time()

        cursor.close()
        connection.close()

        elapsed_time = end_time - start_time
        return True, elapsed_time
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    # 替换为您的数据库连接字符串和要执行的SELECT语句
    connection_string = "udms/admin@192.168.1.15:1521/demodb"
    sql = "SELECT * FROM (SELECT * FROM defect_region) WHERE ROWNUM <= 100000"

    success, elapsed_time = test_network_speed(connection_string, sql)
    
    if success:
        print(f"查询成功！执行时间：{elapsed_time:.4f} 秒")
    else:
        print(f"查询失败：{elapsed_time}")
        
    sql = "SELECT * FROM (SELECT * FROM edc_raw) WHERE ROWNUM <= 100000"

    success, elapsed_time = test_network_speed(connection_string, sql)
    
    if success:
        print(f"查询成功！执行时间：{elapsed_time:.4f} 秒")
    else:
        print(f"查询失败：{elapsed_time}")
