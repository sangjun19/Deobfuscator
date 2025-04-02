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
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 9
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L11
	b	L3
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L130
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L13
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L13
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L13
L19:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L21
L20:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L21:
	b	L13
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L131
L22:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L24:
L131:
	nop
L13:
	b	L130
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L132
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L28
	b	L132
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L26
L29:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L31
L30:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L31:
	b	L26
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L26
L32:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L34
L33:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L34:
	b	L26
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L36
	b	L133
L35:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L37
L36:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L37:
L133:
	nop
L26:
	b	L132
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L134
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L41
	b	L134
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L43
	b	L39
L42:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L44:
	b	L39
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L39
L45:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L47:
	b	L39
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L49
	b	L135
L48:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L50
L49:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L50:
L135:
	nop
L39:
	b	L134
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L54
	b	L136
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L56
	b	L52
L55:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L57:
	b	L52
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L59
	b	L52
L58:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L60
L59:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L60:
	b	L52
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L62
	b	L137
L61:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L63
L62:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L63:
L137:
	nop
L52:
	b	L136
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L67
	b	L138
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L69
	b	L65
L68:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L65
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L65
L71:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L73
L72:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L73:
	b	L65
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L75
	b	L139
L74:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L76
L75:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L76:
L139:
	nop
L65:
	b	L138
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L80
	b	L140
L79:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L82
	b	L78
L81:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L83
L82:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L83:
	b	L78
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L85
	b	L78
L84:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L86
L85:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L86:
	b	L78
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L88
	b	L141
L87:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L89
L88:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L89:
L141:
	nop
L78:
	b	L140
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L90
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L92
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L93
	b	L142
L92:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L94
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L95
	b	L91
L94:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L96
L95:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L96:
	b	L91
L93:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L91
L97:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L99
L98:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L99:
	b	L91
L90:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L101
	b	L143
L100:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L102
L101:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L102:
L143:
	nop
L91:
	b	L142
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L103
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L106
	b	L144
L105:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L108
	b	L104
L107:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L109
L108:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L109:
	b	L104
L106:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L104
L110:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L112
L111:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L112:
	b	L104
L103:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L113
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L114
	b	L145
L113:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L115
L114:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L115:
L145:
	nop
L104:
	b	L144
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L116
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L146
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L118
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L119
	b	L146
L118:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L120
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L121
	b	L117
L120:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L122
L121:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L122:
	b	L117
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L123
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L124
	b	L117
L123:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L125
L124:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L125:
	b	L117
L116:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L126
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L127
	b	L147
L126:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L128
L127:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L128:
L147:
	nop
L117:
	b	L146
L130:
	nop
	b	L3
L132:
	nop
	b	L3
L134:
	nop
	b	L3
L136:
	nop
	b	L3
L138:
	nop
	b	L3
L140:
	nop
	b	L3
L142:
	nop
	b	L3
L144:
	nop
	b	L3
L146:
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
