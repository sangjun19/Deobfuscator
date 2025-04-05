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
	.ascii "1-4-1\0"
	.align	3
lC7:
	.ascii "1-4-2\0"
	.align	3
lC8:
	.ascii "1-5-1\0"
	.align	3
lC9:
	.ascii "1-5-2\0"
	.align	3
lC10:
	.ascii "2-1-1\0"
	.align	3
lC11:
	.ascii "2-1-2\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-3-1\0"
	.align	3
lC15:
	.ascii "2-3-2\0"
	.align	3
lC16:
	.ascii "2-4-1\0"
	.align	3
lC17:
	.ascii "2-4-2\0"
	.align	3
lC18:
	.ascii "2-5-1\0"
	.align	3
lC19:
	.ascii "2-5-2\0"
	.align	3
lC20:
	.ascii "3-1-1\0"
	.align	3
lC21:
	.ascii "3-1-2\0"
	.align	3
lC22:
	.ascii "3-2-1\0"
	.align	3
lC23:
	.ascii "3-2-2\0"
	.align	3
lC24:
	.ascii "3-3-1\0"
	.align	3
lC25:
	.ascii "3-3-2\0"
	.align	3
lC26:
	.ascii "3-4-1\0"
	.align	3
lC27:
	.ascii "3-4-2\0"
	.align	3
lC28:
	.ascii "3-5-1\0"
	.align	3
lC29:
	.ascii "3-5-2\0"
	.align	3
lC30:
	.ascii "4-1-1\0"
	.align	3
lC31:
	.ascii "4-1-2\0"
	.align	3
lC32:
	.ascii "4-2-1\0"
	.align	3
lC33:
	.ascii "4-2-2\0"
	.align	3
lC34:
	.ascii "4-3-1\0"
	.align	3
lC35:
	.ascii "4-3-2\0"
	.align	3
lC36:
	.ascii "4-4-1\0"
	.align	3
lC37:
	.ascii "4-4-2\0"
	.align	3
lC38:
	.ascii "4-5-1\0"
	.align	3
lC39:
	.ascii "4-5-2\0"
	.align	3
lC40:
	.ascii "5-1-1\0"
	.align	3
lC41:
	.ascii "5-1-2\0"
	.align	3
lC42:
	.ascii "5-2-1\0"
	.align	3
lC43:
	.ascii "5-2-2\0"
	.align	3
lC44:
	.ascii "5-3-1\0"
	.align	3
lC45:
	.ascii "5-3-2\0"
	.align	3
lC46:
	.ascii "5-4-1\0"
	.align	3
lC47:
	.ascii "5-4-2\0"
	.align	3
lC48:
	.ascii "5-5-1\0"
	.align	3
lC49:
	.ascii "5-5-2\0"
	.align	3
lC50:
	.ascii "6-1-1\0"
	.align	3
lC51:
	.ascii "6-1-2\0"
	.align	3
lC52:
	.ascii "6-2-1\0"
	.align	3
lC53:
	.ascii "6-2-2\0"
	.align	3
lC54:
	.ascii "6-3-1\0"
	.align	3
lC55:
	.ascii "6-3-2\0"
	.align	3
lC56:
	.ascii "6-4-1\0"
	.align	3
lC57:
	.ascii "6-4-2\0"
	.align	3
lC58:
	.ascii "6-5-1\0"
	.align	3
lC59:
	.ascii "6-5-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 6
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L8
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L136
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L10
L15:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L17:
	b	L10
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L19
	b	L10
L18:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L20
L19:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L20:
	b	L10
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L10
L21:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L23
L22:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L23:
	b	L10
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L10
L24:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L26:
	b	L10
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L137
L27:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L29
L28:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L29:
L137:
	nop
L10:
	b	L136
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L35
	b	L138
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L31
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
	b	L31
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L31
L39:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L41
L40:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L41:
	b	L31
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L43
	b	L31
L42:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L44:
	b	L31
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L31
L45:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L47:
	b	L31
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L49
	b	L139
L48:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L50
L49:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L50:
L139:
	nop
L31:
	b	L138
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L56
	b	L140
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L52
L57:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L59
L58:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L59:
	b	L52
L56:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L61
	b	L52
L60:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L62
L61:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L62:
	b	L52
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L64
	b	L52
L63:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L65
L64:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L65:
	b	L52
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L66
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L67
	b	L52
L66:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L68
L67:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L68:
	b	L52
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L70
	b	L141
L69:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L71
L70:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L71:
L141:
	nop
L52:
	b	L140
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L72
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L77
	b	L142
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L78
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L79
	b	L73
L78:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L80
L79:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L80:
	b	L73
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L82
	b	L73
L81:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L83
L82:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L83:
	b	L73
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L85
	b	L73
L84:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L86
L85:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L86:
	b	L73
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L88
	b	L73
L87:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L89
L88:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L89:
	b	L73
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L91
	b	L143
L90:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L92:
L143:
	nop
L73:
	b	L142
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L93
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L95
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L98
	b	L144
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L100
	b	L94
L99:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L101
L100:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L101:
	b	L94
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L103
	b	L94
L102:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L104
L103:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L104:
	b	L94
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L94
L105:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L107
L106:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L107:
	b	L94
L95:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L109
	b	L94
L108:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L110
L109:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L110:
	b	L94
L93:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L111
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L112
	b	L145
L111:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L113
L112:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L113:
L145:
	nop
L94:
	b	L144
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L114
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L146
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L116
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L146
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L117
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
	b	L115
L120:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L122
L121:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L122:
	b	L115
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L123
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L124
	b	L115
L123:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L125
L124:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L125:
	b	L115
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L126
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L127
	b	L115
L126:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L128
L127:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L128:
	b	L115
L116:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L129
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L130
	b	L115
L129:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L131
L130:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L131:
	b	L115
L114:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L132
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L133
	b	L147
L132:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L134
L133:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L134:
L147:
	nop
L115:
	b	L146
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
