	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-1-2\0"
	.align	3
lC2:
	.ascii "1-2-1\0"
	.align	3
lC3:
	.ascii "1-2-2\0"
	.align	3
lC4:
	.ascii "1-3-1\0"
	.align	3
lC5:
	.ascii "1-3-2\0"
	.align	3
lC6:
	.ascii "2-1-1\0"
	.align	3
lC7:
	.ascii "2-1-2\0"
	.align	3
lC8:
	.ascii "2-2-1\0"
	.align	3
lC9:
	.ascii "2-2-2\0"
	.align	3
lC10:
	.ascii "2-3-1\0"
	.align	3
lC11:
	.ascii "2-3-2\0"
	.align	3
lC12:
	.ascii "3-1-1\0"
	.align	3
lC13:
	.ascii "3-1-2\0"
	.align	3
lC14:
	.ascii "3-2-1\0"
	.align	3
lC15:
	.ascii "3-2-2\0"
	.align	3
lC16:
	.ascii "3-3-1\0"
	.align	3
lC17:
	.ascii "3-3-2\0"
	.align	3
lC18:
	.ascii "4-1-1\0"
	.align	3
lC19:
	.ascii "4-1-2\0"
	.align	3
lC20:
	.ascii "4-2-1\0"
	.align	3
lC21:
	.ascii "4-2-2\0"
	.align	3
lC22:
	.ascii "4-3-1\0"
	.align	3
lC23:
	.ascii "4-3-2\0"
	.align	3
lC24:
	.ascii "5-1-1\0"
	.align	3
lC25:
	.ascii "5-1-2\0"
	.align	3
lC26:
	.ascii "5-2-1\0"
	.align	3
lC27:
	.ascii "5-2-2\0"
	.align	3
lC28:
	.ascii "5-3-1\0"
	.align	3
lC29:
	.ascii "5-3-2\0"
	.align	3
lC30:
	.ascii "6-1-1\0"
	.align	3
lC31:
	.ascii "6-1-2\0"
	.align	3
lC32:
	.ascii "6-2-1\0"
	.align	3
lC33:
	.ascii "6-2-2\0"
	.align	3
lC34:
	.ascii "6-3-1\0"
	.align	3
lC35:
	.ascii "6-3-2\0"
	.align	3
lC36:
	.ascii "7-1-1\0"
	.align	3
lC37:
	.ascii "7-1-2\0"
	.align	3
lC38:
	.ascii "7-2-1\0"
	.align	3
lC39:
	.ascii "7-2-2\0"
	.align	3
lC40:
	.ascii "7-3-1\0"
	.align	3
lC41:
	.ascii "7-3-2\0"
	.align	3
lC42:
	.ascii "8-1-1\0"
	.align	3
lC43:
	.ascii "8-1-2\0"
	.align	3
lC44:
	.ascii "8-2-1\0"
	.align	3
lC45:
	.ascii "8-2-2\0"
	.align	3
lC46:
	.ascii "8-3-1\0"
	.align	3
lC47:
	.ascii "8-3-2\0"
	.align	3
lC48:
	.ascii "9-1-1\0"
	.align	3
lC49:
	.ascii "9-1-2\0"
	.align	3
lC50:
	.ascii "9-2-1\0"
	.align	3
lC51:
	.ascii "9-2-2\0"
	.align	3
lC52:
	.ascii "9-3-1\0"
	.align	3
lC53:
	.ascii "9-3-2\0"
	.align	3
lC54:
	.ascii "10-1-1\0"
	.align	3
lC55:
	.ascii "10-1-2\0"
	.align	3
lC56:
	.ascii "10-2-1\0"
	.align	3
lC57:
	.ascii "10-2-2\0"
	.align	3
lC58:
	.ascii "10-3-1\0"
	.align	3
lC59:
	.ascii "10-3-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 10
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 10
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 10
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L12
	b	L3
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L16
	b	L144
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L14
L17:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L14
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L14
L20:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L14
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L145
L23:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L25:
L145:
	nop
L14:
	b	L144
L12:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L146
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L29
	b	L146
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L31
	b	L27
L30:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L32
L31:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L32:
	b	L27
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L27
L33:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L35:
	b	L27
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L147
L36:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L38:
L147:
	nop
L27:
	b	L146
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L42
	b	L148
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L40
L43:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L45
L44:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L45:
	b	L40
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L40
L46:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L40
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L149
L49:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L51
L50:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L51:
L149:
	nop
L40:
	b	L148
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L150
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L55
	b	L150
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L53
L56:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L58
L57:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L58:
	b	L53
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L53
L59:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L61:
	b	L53
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L151
L62:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L64:
L151:
	nop
L53:
	b	L150
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L152
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L68
	b	L152
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L70
	b	L66
L69:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L71
L70:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L71:
	b	L66
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L72
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L73
	b	L66
L72:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L74
L73:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L74:
	b	L66
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L153
L75:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L77
L76:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L77:
L153:
	nop
L66:
	b	L152
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L81
	b	L154
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L83
	b	L79
L82:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L84
L83:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L84:
	b	L79
L81:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L86
	b	L79
L85:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L87
L86:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L87:
	b	L79
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L89
	b	L155
L88:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L90
L89:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L90:
L155:
	nop
L79:
	b	L154
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L94
	b	L156
L93:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L96
	b	L92
L95:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L97
L96:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L97:
	b	L92
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L98
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L99
	b	L92
L98:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L100
L99:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L100:
	b	L92
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L157
L101:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L103
L102:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L103:
L157:
	nop
L92:
	b	L156
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L104
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L106
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L107
	b	L158
L106:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L109
	b	L105
L108:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L110
L109:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L110:
	b	L105
L107:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L111
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L112
	b	L105
L111:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L113
L112:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L113:
	b	L105
L104:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L115
	b	L159
L114:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L116
L115:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L116:
L159:
	nop
L105:
	b	L158
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L117
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L119
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L120
	b	L160
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L122
	b	L118
L121:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L123
L122:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L123:
	b	L118
L120:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L124
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L125
	b	L118
L124:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L126
L125:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L126:
	b	L118
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L128
	b	L161
L127:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L129
L128:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L129:
L161:
	nop
L118:
	b	L160
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L130
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L132
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L133
	b	L162
L132:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L134
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L135
	b	L131
L134:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L136
L135:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L136:
	b	L131
L133:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L137
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L138
	b	L131
L137:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L139
L138:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L139:
	b	L131
L130:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L140
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L141
	b	L163
L140:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L142
L141:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L142:
L163:
	nop
L131:
	b	L162
L144:
	nop
	b	L3
L146:
	nop
	b	L3
L148:
	nop
	b	L3
L150:
	nop
	b	L3
L152:
	nop
	b	L3
L154:
	nop
	b	L3
L156:
	nop
	b	L3
L158:
	nop
	b	L3
L160:
	nop
	b	L3
L162:
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
