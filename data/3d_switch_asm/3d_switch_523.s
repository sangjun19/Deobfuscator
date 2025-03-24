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
	.ascii "1-1-3\0"
	.align	3
lC3:
	.ascii "1-2-1\0"
	.align	3
lC4:
	.ascii "1-2-2\0"
	.align	3
lC5:
	.ascii "1-2-3\0"
	.align	3
lC6:
	.ascii "1-3-1\0"
	.align	3
lC7:
	.ascii "1-3-2\0"
	.align	3
lC8:
	.ascii "1-3-3\0"
	.align	3
lC9:
	.ascii "2-1-1\0"
	.align	3
lC10:
	.ascii "2-1-2\0"
	.align	3
lC11:
	.ascii "2-1-3\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-2-3\0"
	.align	3
lC15:
	.ascii "2-3-1\0"
	.align	3
lC16:
	.ascii "2-3-2\0"
	.align	3
lC17:
	.ascii "2-3-3\0"
	.align	3
lC18:
	.ascii "3-1-1\0"
	.align	3
lC19:
	.ascii "3-1-2\0"
	.align	3
lC20:
	.ascii "3-1-3\0"
	.align	3
lC21:
	.ascii "3-2-1\0"
	.align	3
lC22:
	.ascii "3-2-2\0"
	.align	3
lC23:
	.ascii "3-2-3\0"
	.align	3
lC24:
	.ascii "3-3-1\0"
	.align	3
lC25:
	.ascii "3-3-2\0"
	.align	3
lC26:
	.ascii "3-3-3\0"
	.align	3
lC27:
	.ascii "4-1-1\0"
	.align	3
lC28:
	.ascii "4-1-2\0"
	.align	3
lC29:
	.ascii "4-1-3\0"
	.align	3
lC30:
	.ascii "4-2-1\0"
	.align	3
lC31:
	.ascii "4-2-2\0"
	.align	3
lC32:
	.ascii "4-2-3\0"
	.align	3
lC33:
	.ascii "4-3-1\0"
	.align	3
lC34:
	.ascii "4-3-2\0"
	.align	3
lC35:
	.ascii "4-3-3\0"
	.align	3
lC36:
	.ascii "5-1-1\0"
	.align	3
lC37:
	.ascii "5-1-2\0"
	.align	3
lC38:
	.ascii "5-1-3\0"
	.align	3
lC39:
	.ascii "5-2-1\0"
	.align	3
lC40:
	.ascii "5-2-2\0"
	.align	3
lC41:
	.ascii "5-2-3\0"
	.align	3
lC42:
	.ascii "5-3-1\0"
	.align	3
lC43:
	.ascii "5-3-2\0"
	.align	3
lC44:
	.ascii "5-3-3\0"
	.align	3
lC45:
	.ascii "6-1-1\0"
	.align	3
lC46:
	.ascii "6-1-2\0"
	.align	3
lC47:
	.ascii "6-1-3\0"
	.align	3
lC48:
	.ascii "6-2-1\0"
	.align	3
lC49:
	.ascii "6-2-2\0"
	.align	3
lC50:
	.ascii "6-2-3\0"
	.align	3
lC51:
	.ascii "6-3-1\0"
	.align	3
lC52:
	.ascii "6-3-2\0"
	.align	3
lC53:
	.ascii "6-3-3\0"
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
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 3
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
	cmp	w0, 3
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L106
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L12
	b	L106
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L107
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L107
L15:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L14
L16:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L14
L13:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
L14:
	b	L107
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L108
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L108
L19:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L18
L20:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L108
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L109
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L109
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L22
L24:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L109
L107:
	nop
	b	L106
L108:
	nop
	b	L106
L109:
	nop
	b	L106
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L110
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L28
	b	L110
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L111
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L111
L31:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L30
L32:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L111
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L112
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L36
	b	L112
L35:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L34
L36:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L34
L33:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L34:
	b	L112
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L113
L39:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L38
L40:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L38:
	b	L113
L111:
	nop
	b	L110
L112:
	nop
	b	L110
L113:
	nop
	b	L110
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L44
	b	L114
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L48
	b	L115
L47:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L46
L48:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L46
L45:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L46:
	b	L115
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L116
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L52
	b	L116
L51:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L50
L52:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L50
L49:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L50:
	b	L116
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L117
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L56
	b	L117
L55:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L54
L56:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L54
L53:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L54:
	b	L117
L115:
	nop
	b	L114
L116:
	nop
	b	L114
L117:
	nop
	b	L114
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L57
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L60
	b	L118
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L119
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L64
	b	L119
L63:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L62
L64:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L62
L61:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L62:
	b	L119
L60:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L120
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L68
	b	L120
L67:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L66
L68:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L66
L65:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
L66:
	b	L120
L57:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L121
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L121
L71:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L70
L72:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L121
L119:
	nop
	b	L118
L120:
	nop
	b	L118
L121:
	nop
	b	L118
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L73
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L76
	b	L122
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L123
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L80
	b	L123
L79:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L78
L80:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L78
L77:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
L78:
	b	L123
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L124
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L84
	b	L124
L83:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	b	L82
L84:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L124
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L125
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L88
	b	L125
L87:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L86
L88:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L86
L85:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L86:
	b	L125
L123:
	nop
	b	L122
L124:
	nop
	b	L122
L125:
	nop
	b	L122
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L89
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L126
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L92
	b	L126
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L127
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L96
	b	L127
L95:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L94
L96:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L94
L93:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L94:
	b	L127
L92:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L128
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L100
	b	L128
L99:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L98
L100:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L98
L97:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	nop
L98:
	b	L128
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L104
	b	L129
L103:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	b	L102
L104:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L102
L101:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L102:
	b	L129
L127:
	nop
	b	L126
L128:
	nop
	b	L126
L129:
	nop
	b	L126
L106:
	nop
	b	L3
L110:
	nop
	b	L3
L114:
	nop
	b	L3
L118:
	nop
	b	L3
L122:
	nop
	b	L3
L126:
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
