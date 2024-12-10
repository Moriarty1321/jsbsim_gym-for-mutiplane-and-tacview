from socket import *
import time

IP = 'localhost'
port = 7788

client=socket(AF_INET, SOCK_STREAM)
client.bind((IP,port))
client.listen(5)

print(f'Server start at {IP}:{port}')
print('waiting for connection...')

record = ''

while True:
    print('waiting??')
    conn, addr = client.accept()
    print(f'Connected by {addr}')
    
    info = 'XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nsimulator\n\0'.encode('utf-8')
    conn.send(info)
    
    data = conn.recv(1024)
    print(data.decode('utf-8'))
    
    init_info = 'FileType=text/acmi/tacview\nFileVersion=2.2\n'
    conn.send(init_info.encode('utf-8'))
    record+=init_info
    
    
    reset_info = '0,ReferenceLongitude=120\n0,ReferenceLatitude=25\n0,ReferenceTime=2022-07-06T08:39:21.153Z\n'
    conn.send(reset_info.encode('utf-8'))
    record+=reset_info
    
    state = '#0\n101,T=120|25|2000|0|0|0,Type=Air+FixedWing,Name=GJ-11,Color=Blue\n'
    conn.send(state.encode('utf-8'))
    record+=state
    
    for i in range(1000):
        #print(f'sending data {i}')
        t = '#'+str(0.2*(i+1))+'\n'
        latitude = str(25+i*0.0001)
        state = t+'101,T=120|'+latitude+'|2000|0|0|0\n'
        conn.send(state.encode('utf-8'))
        #time.sleep(0.1)
    print('done and ?')
    print(record)
    
    
        
    
    
        