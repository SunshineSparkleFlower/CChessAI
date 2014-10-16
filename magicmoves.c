/**
 *magicmoves.h
 *
 *Source file for magic move bitboard generation.
 *
 *See header file for instructions on usage.
 *
 *The magic keys are not optimal for all squares but they are very close
 *to optimal.
 *
 *Copyright (C) 2007 Pradyumna Kannan.
 *
 *This code is provided 'as-is', without any express or implied warranty.
 *In no event will the authors be held liable for any damages arising from
 *the use of this code. Permission is granted to anyone to use this
 *code for any purpose, including commercial applications, and to alter
 *it and redistribute it freely, subject to the following restrictions:
 *
 *1. The origin of this code must not be misrepresented; you must not
 *claim that you wrote the original code. If you use this code in a
 *product, an acknowledgment in the product documentation would be
 *appreciated but is not required.
 *
 *2. Altered source versions must be plainly marked as such, and must not be
 *misrepresented as being the original code.
 *
 *3. This notice may not be removed or altered from any source distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bitboard.h"
#include "magicmoves.h"


const unsigned int magicmoves_r_shift[64]=
{
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 53, 53, 53, 53, 53
};

const u64 magicmoves_r_magics[64]=
{
    C64(0x0080001020400080), C64(0x0040001000200040), C64(0x0080081000200080), C64(0x0080040800100080),
    C64(0x0080020400080080), C64(0x0080010200040080), C64(0x0080008001000200), C64(0x0080002040800100),
    C64(0x0000800020400080), C64(0x0000400020005000), C64(0x0000801000200080), C64(0x0000800800100080),
    C64(0x0000800400080080), C64(0x0000800200040080), C64(0x0000800100020080), C64(0x0000800040800100),
    C64(0x0000208000400080), C64(0x0000404000201000), C64(0x0000808010002000), C64(0x0000808008001000),
    C64(0x0000808004000800), C64(0x0000808002000400), C64(0x0000010100020004), C64(0x0000020000408104),
    C64(0x0000208080004000), C64(0x0000200040005000), C64(0x0000100080200080), C64(0x0000080080100080),
    C64(0x0000040080080080), C64(0x0000020080040080), C64(0x0000010080800200), C64(0x0000800080004100),
    C64(0x0000204000800080), C64(0x0000200040401000), C64(0x0000100080802000), C64(0x0000080080801000),
    C64(0x0000040080800800), C64(0x0000020080800400), C64(0x0000020001010004), C64(0x0000800040800100),
    C64(0x0000204000808000), C64(0x0000200040008080), C64(0x0000100020008080), C64(0x0000080010008080),
    C64(0x0000040008008080), C64(0x0000020004008080), C64(0x0000010002008080), C64(0x0000004081020004),
    C64(0x0000204000800080), C64(0x0000200040008080), C64(0x0000100020008080), C64(0x0000080010008080),
    C64(0x0000040008008080), C64(0x0000020004008080), C64(0x0000800100020080), C64(0x0000800041000080),
    C64(0x00FFFCDDFCED714A), C64(0x007FFCDDFCED714A), C64(0x003FFFCDFFD88096), C64(0x0000040810002101),
    C64(0x0001000204080011), C64(0x0001000204000801), C64(0x0001000082000401), C64(0x0001FFFAABFAD1A2)
};

const u64 magicmoves_r_mask[64]=
{	
    C64(0x000101010101017E), C64(0x000202020202027C), C64(0x000404040404047A), C64(0x0008080808080876),
    C64(0x001010101010106E), C64(0x002020202020205E), C64(0x004040404040403E), C64(0x008080808080807E),
    C64(0x0001010101017E00), C64(0x0002020202027C00), C64(0x0004040404047A00), C64(0x0008080808087600),
    C64(0x0010101010106E00), C64(0x0020202020205E00), C64(0x0040404040403E00), C64(0x0080808080807E00),
    C64(0x00010101017E0100), C64(0x00020202027C0200), C64(0x00040404047A0400), C64(0x0008080808760800),
    C64(0x00101010106E1000), C64(0x00202020205E2000), C64(0x00404040403E4000), C64(0x00808080807E8000),
    C64(0x000101017E010100), C64(0x000202027C020200), C64(0x000404047A040400), C64(0x0008080876080800),
    C64(0x001010106E101000), C64(0x002020205E202000), C64(0x004040403E404000), C64(0x008080807E808000),
    C64(0x0001017E01010100), C64(0x0002027C02020200), C64(0x0004047A04040400), C64(0x0008087608080800),
    C64(0x0010106E10101000), C64(0x0020205E20202000), C64(0x0040403E40404000), C64(0x0080807E80808000),
    C64(0x00017E0101010100), C64(0x00027C0202020200), C64(0x00047A0404040400), C64(0x0008760808080800),
    C64(0x00106E1010101000), C64(0x00205E2020202000), C64(0x00403E4040404000), C64(0x00807E8080808000),
    C64(0x007E010101010100), C64(0x007C020202020200), C64(0x007A040404040400), C64(0x0076080808080800),
    C64(0x006E101010101000), C64(0x005E202020202000), C64(0x003E404040404000), C64(0x007E808080808000),
    C64(0x7E01010101010100), C64(0x7C02020202020200), C64(0x7A04040404040400), C64(0x7608080808080800),
    C64(0x6E10101010101000), C64(0x5E20202020202000), C64(0x3E40404040404000), C64(0x7E80808080808000)
};

//my original tables for bishops
const unsigned int magicmoves_b_shift[64]=
{
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

const u64 magicmoves_b_magics[64]=
{
    C64(0x0002020202020200), C64(0x0002020202020000), C64(0x0004010202000000), C64(0x0004040080000000),
    C64(0x0001104000000000), C64(0x0000821040000000), C64(0x0000410410400000), C64(0x0000104104104000),
    C64(0x0000040404040400), C64(0x0000020202020200), C64(0x0000040102020000), C64(0x0000040400800000),
    C64(0x0000011040000000), C64(0x0000008210400000), C64(0x0000004104104000), C64(0x0000002082082000),
    C64(0x0004000808080800), C64(0x0002000404040400), C64(0x0001000202020200), C64(0x0000800802004000),
    C64(0x0000800400A00000), C64(0x0000200100884000), C64(0x0000400082082000), C64(0x0000200041041000),
    C64(0x0002080010101000), C64(0x0001040008080800), C64(0x0000208004010400), C64(0x0000404004010200),
    C64(0x0000840000802000), C64(0x0000404002011000), C64(0x0000808001041000), C64(0x0000404000820800),
    C64(0x0001041000202000), C64(0x0000820800101000), C64(0x0000104400080800), C64(0x0000020080080080),
    C64(0x0000404040040100), C64(0x0000808100020100), C64(0x0001010100020800), C64(0x0000808080010400),
    C64(0x0000820820004000), C64(0x0000410410002000), C64(0x0000082088001000), C64(0x0000002011000800),
    C64(0x0000080100400400), C64(0x0001010101000200), C64(0x0002020202000400), C64(0x0001010101000200),
    C64(0x0000410410400000), C64(0x0000208208200000), C64(0x0000002084100000), C64(0x0000000020880000),
    C64(0x0000001002020000), C64(0x0000040408020000), C64(0x0004040404040000), C64(0x0002020202020000),
    C64(0x0000104104104000), C64(0x0000002082082000), C64(0x0000000020841000), C64(0x0000000000208800),
    C64(0x0000000010020200), C64(0x0000000404080200), C64(0x0000040404040400), C64(0x0002020202020200)
};


const u64 magicmoves_b_mask[64]=
{
    C64(0x0040201008040200), C64(0x0000402010080400), C64(0x0000004020100A00), C64(0x0000000040221400),
    C64(0x0000000002442800), C64(0x0000000204085000), C64(0x0000020408102000), C64(0x0002040810204000),
    C64(0x0020100804020000), C64(0x0040201008040000), C64(0x00004020100A0000), C64(0x0000004022140000),
    C64(0x0000000244280000), C64(0x0000020408500000), C64(0x0002040810200000), C64(0x0004081020400000),
    C64(0x0010080402000200), C64(0x0020100804000400), C64(0x004020100A000A00), C64(0x0000402214001400),
    C64(0x0000024428002800), C64(0x0002040850005000), C64(0x0004081020002000), C64(0x0008102040004000),
    C64(0x0008040200020400), C64(0x0010080400040800), C64(0x0020100A000A1000), C64(0x0040221400142200),
    C64(0x0002442800284400), C64(0x0004085000500800), C64(0x0008102000201000), C64(0x0010204000402000),
    C64(0x0004020002040800), C64(0x0008040004081000), C64(0x00100A000A102000), C64(0x0022140014224000),
    C64(0x0044280028440200), C64(0x0008500050080400), C64(0x0010200020100800), C64(0x0020400040201000),
    C64(0x0002000204081000), C64(0x0004000408102000), C64(0x000A000A10204000), C64(0x0014001422400000),
    C64(0x0028002844020000), C64(0x0050005008040200), C64(0x0020002010080400), C64(0x0040004020100800),
    C64(0x0000020408102000), C64(0x0000040810204000), C64(0x00000A1020400000), C64(0x0000142240000000),
    C64(0x0000284402000000), C64(0x0000500804020000), C64(0x0000201008040200), C64(0x0000402010080400),
    C64(0x0002040810204000), C64(0x0004081020400000), C64(0x000A102040000000), C64(0x0014224000000000),
    C64(0x0028440200000000), C64(0x0050080402000000), C64(0x0020100804020000), C64(0x0040201008040200)
};

u64 magicmovesbdb[64][1<<9];

u64 magicmovesrdb[64][1<<12];

u64 initmagicmoves_occ(const int* squares, const int numSquares, const u64 linocc)
{
    int i;
    u64 ret=0;
    for(i=0;i<numSquares;i++)
        if(linocc&(((u64)(1))<<i)) ret|=(((u64)(1))<<squares[i]);
    return ret;
}

u64 initmagicmoves_Rmoves(const int square, const u64 occ)
{
    u64 ret=0;
    u64 bit;
    u64 rowbits=(((u64)0xFF)<<(8*(square/8)));

    bit=(((u64)(1))<<square);
    do
    {
        bit<<=8;
        ret|=bit;
    }while(bit && !(bit&occ));
    bit=(((u64)(1))<<square);
    do
    {
        bit>>=8;
        ret|=bit;
    }while(bit && !(bit&occ));
    bit=(((u64)(1))<<square);
    do
    {
        bit<<=1;
        if(bit&rowbits) ret|=bit;
        else break;
    }while(!(bit&occ));
    bit=(((u64)(1))<<square);
    do
    {
        bit>>=1;
        if(bit&rowbits) ret|=bit;
        else break;
    }while(!(bit&occ));
    return ret;
}

u64 initmagicmoves_Bmoves(const int square, const u64 occ)
{
    u64 ret=0;
    u64 bit;
    u64 bit2;
    u64 rowbits=(((u64)0xFF)<<(8*(square/8)));

    bit=(((u64)(1))<<square);
    bit2=bit;
    do
    {
        bit<<=8-1;
        bit2>>=1;
        if(bit2&rowbits) ret|=bit;
        else break;
    }while(bit && !(bit&occ));
    bit=(((u64)(1))<<square);
    bit2=bit;
    do
    {
        bit<<=8+1;
        bit2<<=1;
        if(bit2&rowbits) ret|=bit;
        else break;
    }while(bit && !(bit&occ));
    bit=(((u64)(1))<<square);
    bit2=bit;
    do
    {
        bit>>=8-1;
        bit2<<=1;
        if(bit2&rowbits) ret|=bit;
        else break;
    }while(bit && !(bit&occ));
    bit=(((u64)(1))<<square);
    bit2=bit;
    do
    {
        bit>>=8+1;
        bit2>>=1;
        if(bit2&rowbits) ret|=bit;
        else break;
    }while(bit && !(bit&occ));
    return ret;
}

//used so that the original indices can be left as const so that the compiler can optimize better
#define BmagicNOMASK2(square, occupancy) \
    magicmovesbdb[square][((occupancy)*magicmoves_b_magics[square])>>MINIMAL_B_BITS_SHIFT(square)]
#define RmagicNOMASK2(square, occupancy) \
    magicmovesrdb[square][((occupancy)*magicmoves_r_magics[square])>>MINIMAL_R_BITS_SHIFT(square)]

void init_magicmoves(void)
{
    int i;

    //for bitscans :
    //initmagicmoves_bitpos64_database[(x*C64(0x07EDD5E59A4E28C2))>>58]
    int initmagicmoves_bitpos64_database[64]={
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5};

    for(i=0;i<64;i++) {
        int squares[64];
        int numsquares=0;
        u64 temp=magicmoves_b_mask[i];
        while(temp) {
            u64 bit=temp&-temp;
            squares[numsquares++]=initmagicmoves_bitpos64_database[(bit*C64(0x07EDD5E59A4E28C2))>>58];
            temp^=bit;
        }

        for(temp=0;temp<(((u64)(1))<<numsquares);temp++) {
            u64 tempocc=initmagicmoves_occ(squares,numsquares,temp);
            BmagicNOMASK2(i,tempocc)=initmagicmoves_Bmoves(i,tempocc);
        }
    }

    for(i=0;i<64;i++) {
        int squares[64];
        int numsquares=0;
        u64 temp=magicmoves_r_mask[i];

        while(temp) {
            u64 bit=temp&-temp;
            squares[numsquares++]=initmagicmoves_bitpos64_database[(bit*C64(0x07EDD5E59A4E28C2))>>58];
            temp^=bit;
        }

        for(temp=0;temp<(((u64)(1))<<numsquares);temp++) {
            u64 tempocc=initmagicmoves_occ(squares,numsquares,temp);
            RmagicNOMASK2(i,tempocc)=initmagicmoves_Rmoves(i,tempocc);
        }
    }
}
