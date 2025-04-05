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
	.ascii "1-6-1\0"
	.align	3
lC11:
	.ascii "1-6-2\0"
	.align	3
lC12:
	.ascii "2-1-1\0"
	.align	3
lC13:
	.ascii "2-1-2\0"
	.align	3
lC14:
	.ascii "2-2-1\0"
	.align	3
lC15:
	.ascii "2-2-2\0"
	.align	3
lC16:
	.ascii "2-3-1\0"
	.align	3
lC17:
	.ascii "2-3-2\0"
	.align	3
lC18:
	.ascii "2-4-1\0"
	.align	3
lC19:
	.ascii "2-4-2\0"
	.align	3
lC20:
	.ascii "2-5-1\0"
	.align	3
lC21:
	.ascii "2-5-2\0"
	.align	3
lC22:
	.ascii "2-6-1\0"
	.align	3
lC23:
	.ascii "2-6-2\0"
	.align	3
lC24:
	.ascii "3-1-1\0"
	.align	3
lC25:
	.ascii "3-1-2\0"
	.align	3
lC26:
	.ascii "3-2-1\0"
	.align	3
lC27:
	.ascii "3-2-2\0"
	.align	3
lC28:
	.ascii "3-3-1\0"
	.align	3
lC29:
	.ascii "3-3-2\0"
	.align	3
lC30:
	.ascii "3-4-1\0"
	.align	3
lC31:
	.ascii "3-4-2\0"
	.align	3
lC32:
	.ascii "3-5-1\0"
	.align	3
lC33:
	.ascii "3-5-2\0"
	.align	3
lC34:
	.ascii "3-6-1\0"
	.align	3
lC35:
	.ascii "3-6-2\0"
	.align	3
lC36:
	.ascii "4-1-1\0"
	.align	3
lC37:
	.ascii "4-1-2\0"
	.align	3
lC38:
	.ascii "4-2-1\0"
	.align	3
lC39:
	.ascii "4-2-2\0"
	.align	3
lC40:
	.ascii "4-3-1\0"
	.align	3
lC41:
	.ascii "4-3-2\0"
	.align	3
lC42:
	.ascii "4-4-1\0"
	.align	3
lC43:
	.ascii "4-4-2\0"
	.align	3
lC44:
	.ascii "4-5-1\0"
	.align	3
lC45:
	.ascii "4-5-2\0"
	.align	3
lC46:
	.ascii "4-6-1\0"
	.align	3
lC47:
	.ascii "4-6-2\0"
	.align	3
lC48:
	.ascii "5-1-1\0"
	.align	3
lC49:
	.ascii "5-1-2\0"
	.align	3
lC50:
	.ascii "5-2-1\0"
	.align	3
lC51:
	.ascii "5-2-2\0"
	.align	3
lC52:
	.ascii "5-3-1\0"
	.align	3
lC53:
	.ascii "5-3-2\0"
	.align	3
lC54:
	.ascii "5-4-1\0"
	.align	3
lC55:
	.ascii "5-4-2\0"
	.align	3
lC56:
	.ascii "5-5-1\0"
	.align	3
lC57:
	.ascii "5-5-2\0"
	.align	3
lC58:
	.ascii "5-6-1\0"
	.align	3
lC59:
	.ascii "5-6-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 5
	str	w0, [x29, 28]
	mov	w0, 6
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L7
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L134
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L134
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L134
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L134
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L134
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L9
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
	b	L9
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L19
	b	L9
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
	b	L9
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L9
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
	b	L9
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L9
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
	b	L9
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L9
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
	b	L9
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L31
	b	L135
L30:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L32
L31:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L32:
L135:
	nop
L9:
	b	L134
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L136
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L39
	b	L136
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L41
	b	L34
L40:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L42:
	b	L34
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L34
L43:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L45
L44:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L45:
	b	L34
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L34
L46:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L34
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L34
L49:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L51
L50:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L51:
	b	L34
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L52
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L53
	b	L34
L52:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L54
L53:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L54:
	b	L34
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L56
	b	L137
L55:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L57:
L137:
	nop
L34:
	b	L136
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L58
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L60
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L61
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L62
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L138
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L64
	b	L138
L63:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L59
L65:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L67
L66:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L67:
	b	L59
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L69
	b	L59
L68:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L59
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L59
L71:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L73
L72:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L73:
	b	L59
L61:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L75
	b	L59
L74:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L76
L75:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L76:
	b	L59
L60:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L78
	b	L59
L77:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L79
L78:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L79:
	b	L59
L58:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L81
	b	L139
L80:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L82:
L139:
	nop
L59:
	b	L138
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L83
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L85
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L86
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L87
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L89
	b	L140
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L91
	b	L84
L90:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L92:
	b	L84
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L94
	b	L84
L93:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L95
L94:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L95:
	b	L84
L87:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L96
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L97
	b	L84
L96:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L98
L97:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L98:
	b	L84
L86:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L100
	b	L84
L99:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L101
L100:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L101:
	b	L84
L85:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L103
	b	L84
L102:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L104
L103:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L104:
	b	L84
L83:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L141
L105:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L107
L106:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L107:
L141:
	nop
L84:
	b	L140
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L108
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L110
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L111
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L112
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L142
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L113
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L114
	b	L142
L113:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L115
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L116
	b	L109
L115:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L117
L116:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L117:
	b	L109
L114:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L118
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L119
	b	L109
L118:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L120
L119:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L120:
	b	L109
L112:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L122
	b	L109
L121:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L123
L122:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L123:
	b	L109
L111:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L124
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L125
	b	L109
L124:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L126
L125:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L126:
	b	L109
L110:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L128
	b	L109
L127:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L129
L128:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L129:
	b	L109
L108:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L130
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L131
	b	L143
L130:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L132
L131:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L132:
L143:
	nop
L109:
	b	L142
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
