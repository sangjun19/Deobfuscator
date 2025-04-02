	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-2-1\0"
	.align	3
lC2:
	.ascii "1-3-1\0"
	.align	3
lC3:
	.ascii "1-4-1\0"
	.align	3
lC4:
	.ascii "1-5-1\0"
	.align	3
lC5:
	.ascii "1-6-1\0"
	.align	3
lC6:
	.ascii "1-7-1\0"
	.align	3
lC7:
	.ascii "1-8-1\0"
	.align	3
lC8:
	.ascii "1-9-1\0"
	.align	3
lC9:
	.ascii "1-10-1\0"
	.align	3
lC10:
	.ascii "2-1-1\0"
	.align	3
lC11:
	.ascii "2-2-1\0"
	.align	3
lC12:
	.ascii "2-3-1\0"
	.align	3
lC13:
	.ascii "2-4-1\0"
	.align	3
lC14:
	.ascii "2-5-1\0"
	.align	3
lC15:
	.ascii "2-6-1\0"
	.align	3
lC16:
	.ascii "2-7-1\0"
	.align	3
lC17:
	.ascii "2-8-1\0"
	.align	3
lC18:
	.ascii "2-9-1\0"
	.align	3
lC19:
	.ascii "2-10-1\0"
	.align	3
lC20:
	.ascii "3-1-1\0"
	.align	3
lC21:
	.ascii "3-2-1\0"
	.align	3
lC22:
	.ascii "3-3-1\0"
	.align	3
lC23:
	.ascii "3-4-1\0"
	.align	3
lC24:
	.ascii "3-5-1\0"
	.align	3
lC25:
	.ascii "3-6-1\0"
	.align	3
lC26:
	.ascii "3-7-1\0"
	.align	3
lC27:
	.ascii "3-8-1\0"
	.align	3
lC28:
	.ascii "3-9-1\0"
	.align	3
lC29:
	.ascii "3-10-1\0"
	.align	3
lC30:
	.ascii "4-1-1\0"
	.align	3
lC31:
	.ascii "4-2-1\0"
	.align	3
lC32:
	.ascii "4-3-1\0"
	.align	3
lC33:
	.ascii "4-4-1\0"
	.align	3
lC34:
	.ascii "4-5-1\0"
	.align	3
lC35:
	.ascii "4-6-1\0"
	.align	3
lC36:
	.ascii "4-7-1\0"
	.align	3
lC37:
	.ascii "4-8-1\0"
	.align	3
lC38:
	.ascii "4-9-1\0"
	.align	3
lC39:
	.ascii "4-10-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 4
	str	w0, [x29, 28]
	mov	w0, 10
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L6
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 10
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 10
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L92
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L17
	b	L92
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L93
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L93
L17:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L94
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L94
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L95
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L95
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L96
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L96
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L97
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L97
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L98
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L98
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L99
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L99
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L100
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L100
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L101
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L101
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L102
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L102
L93:
	nop
	b	L92
L94:
	nop
	b	L92
L95:
	nop
	b	L92
L96:
	nop
	b	L92
L97:
	nop
	b	L92
L98:
	nop
	b	L92
L99:
	nop
	b	L92
L100:
	nop
	b	L92
L101:
	nop
	b	L92
L102:
	nop
	b	L92
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 10
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 10
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L103
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L38
	b	L103
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L104
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L104
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L105
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L105
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L106
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L106
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L107
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L107
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L108
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L108
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L109
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L109
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L110
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L110
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L111
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L111
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L112
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L112
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L113
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L113
L104:
	nop
	b	L103
L105:
	nop
	b	L103
L106:
	nop
	b	L103
L107:
	nop
	b	L103
L108:
	nop
	b	L103
L109:
	nop
	b	L103
L110:
	nop
	b	L103
L111:
	nop
	b	L103
L112:
	nop
	b	L103
L113:
	nop
	b	L103
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 10
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 10
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L56
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L57
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L59
	b	L114
L58:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L115
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L115
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L116
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L116
L57:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L117
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L117
L56:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L118
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L118
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L119
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L119
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L120
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L120
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L121
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L121
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L122
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L122
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L123
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L123
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L124
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L124
L115:
	nop
	b	L114
L116:
	nop
	b	L114
L117:
	nop
	b	L114
L118:
	nop
	b	L114
L119:
	nop
	b	L114
L120:
	nop
	b	L114
L121:
	nop
	b	L114
L122:
	nop
	b	L114
L123:
	nop
	b	L114
L124:
	nop
	b	L114
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 10
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 10
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L72
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L73
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L125
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L80
	b	L125
L79:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L126
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	nop
	b	L126
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L127
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
	b	L127
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L128
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
	b	L128
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L129
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
	b	L129
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L130
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
	b	L130
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L131
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
	b	L131
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L132
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	nop
	b	L132
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L133
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
	b	L133
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L134
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
	b	L134
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L135
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
	b	L135
L126:
	nop
	b	L125
L127:
	nop
	b	L125
L128:
	nop
	b	L125
L129:
	nop
	b	L125
L130:
	nop
	b	L125
L131:
	nop
	b	L125
L132:
	nop
	b	L125
L133:
	nop
	b	L125
L134:
	nop
	b	L125
L135:
	nop
	b	L125
L92:
	nop
	b	L3
L103:
	nop
	b	L3
L114:
	nop
	b	L3
L125:
	nop
L3:
	mov	w0, 0
	ldp	x29, x30, [sp], 32
LCFI2:
	ret
LFE1:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.uleb128 0
	.byte	0x4
	.set L$set$3,LCFI0-LFB1
	.long L$set$3
	.byte	0xe
	.uleb128 0x20
	.byte	0x9d
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x3
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE1:
	.ident	"GCC: (Homebrew GCC 14.2.0_1) 14.2.0"
	.subsections_via_symbols
