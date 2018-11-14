from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import mysql.connector
from mysql.connector import Error

HOST = 'your-host-ip'
DATABASE = 'your-database-name'
USER = 'your-user-name'
PASSWORD = 'your-password'


class MySQLWriter(object):
    def __init__(self):
        self._connect()

    def _connect(self):
        try:
            self.connection = mysql.connector.connect(host=HOST,
                                                      database=DATABASE,
                                                      user=USER,
                                                      password=PASSWORD)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
        except Error as e:
            print("Error while connecting to MySQL", e)

    def reconnect(self):
        if not self.connection.is_connected():
            self._connect()

    def close(self):
        if self.connection.is_connected():
            self.cursor.close()
            self.connection.close()

    def write(self, table, fields, rows):
        stmt = "INSERT INTO %s (%s) VALUES (%s)" % (table, ','.join(fields), ','.join(['%s'] * len(fields)))
        self.cursor.executemany(stmt, rows)
        self.connection.commit()


def write_rows(rows, table_name, field_names):
    mysql_writer = MySQLWriter()
    mysql_writer.write(table_name, field_names, rows)
    mysql_writer.close()