#! /usr/bin/env python
"""
 Project: Python Chess
 File name: ChessPlayer_remote.py
 Description:  Stores info on remote chess player.
    
 Copyright (C) 2009 Steve Osborne, srosborne (at) gmail.com
 http://yakinikuman.wordpress.com/
"""

import socket
import json

class ChessPlayer_remote:
    s = None
    def __init__(self, player1, remoteHost):
        try:
            player1.color = "black"
            self.color = "white"
            self.s = socket.create_connection((remoteHost, 4444))
        except Exception:
            print "Failed to connect to remote host"
            pass

        if not self.s:
            player1.color = "white"
            self.color = "black"

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 4444)) 
            s.listen(1) 

            print "Waiting for remote host"
            self.s, address = s.accept() 
            print "connected to", address
            s.close()


        self.s.send(player1.GetName())

        self.name = self.s.recv(128)
        self.type = 'remote'
        
    def GetName(self):
        return self.name
        
    def GetColor(self):
        return self.color
    
    def GetType(self):
        return self.type

    def GetMove(self):
        move = self.s.recv(512)
        return json.loads(move)

    def SendMove(self, moveTuple):
        self.s.send(json.dumps(moveTuple))
        
if __name__ == "__main__":
    
    p = ChessPlayer_remote("localhost")

    if p.s:
        print p.GetName()
        print p.GetColor()
        print p.GetType()
