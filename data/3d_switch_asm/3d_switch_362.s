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
	.ascii "1-7-1\0"
	.align	3
lC13:
	.ascii "1-7-2\0"
	.align	3
lC14:
	.ascii "2-1-1\0"
	.align	3
lC15:
	.ascii "2-1-2\0"
	.align	3
lC16:
	.ascii "2-2-1\0"
	.align	3
lC17:
	.ascii "2-2-2\0"
	.align	3
lC18:
	.ascii "2-3-1\0"
	.align	3
lC19:
	.ascii "2-3-2\0"
	.align	3
lC20:
	.ascii "2-4-1\0"
	.align	3
lC21:
	.ascii "2-4-2\0"
	.align	3
lC22:
	.ascii "2-5-1\0"
	.align	3
lC23:
	.ascii "2-5-2\0"
	.align	3
lC24:
	.ascii "2-6-1\0"
	.align	3
lC25:
	.ascii "2-6-2\0"
	.align	3
lC26:
	.ascii "2-7-1\0"
	.align	3
lC27:
	.ascii "2-7-2\0"
	.align	3
lC28:
	.ascii "3-1-1\0"
	.align	3
lC29:
	.ascii "3-1-2\0"
	.align	3
lC30:
	.ascii "3-2-1\0"
	.align	3
lC31:
	.ascii "3-2-2\0"
	.align	3
lC32:
	.ascii "3-3-1\0"
	.align	3
lC33:
	.ascii "3-3-2\0"
	.align	3
lC34:
	.ascii "3-4-1\0"
	.align	3
lC35:
	.ascii "3-4-2\0"
	.align	3
lC36:
	.ascii "3-5-1\0"
	.align	3
lC37:
	.ascii "3-5-2\0"
	.align	3
lC38:
	.ascii "3-6-1\0"
	.align	3
lC39:
	.ascii "3-6-2\0"
	.align	3
lC40:
	.ascii "3-7-1\0"
	.align	3
lC41:
	.ascii "3-7-2\0"
	.align	3
lC42:
	.ascii "4-1-1\0"
	.align	3
lC43:
	.ascii "4-1-2\0"
	.align	3
lC44:
	.ascii "4-2-1\0"
	.align	3
lC45:
	.ascii "4-2-2\0"
	.align	3
lC46:
	.ascii "4-3-1\0"
	.align	3
lC47:
	.ascii "4-3-2\0"
	.align	3
lC48:
	.ascii "4-4-1\0"
	.align	3
lC49:
	.ascii "4-4-2\0"
	.align	3
lC50:
	.ascii "4-5-1\0"
	.align	3
lC51:
	.ascii "4-5-2\0"
	.align	3
lC52:
	.ascii "4-6-1\0"
	.align	3
lC53:
	.ascii "4-6-2\0"
	.align	3
lC54:
	.ascii "4-7-1\0"
	.align	3
lC55:
	.ascii "4-7-2\0"
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
	mov	w0, 7
	str	w0, [x29, 24]
	mov	w0, 2
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
	cmp	w0, 7
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L124
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L8
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
	b	L8
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L19
	b	L8
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
	b	L8
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L8
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
	b	L8
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L8
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
	b	L8
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L8
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
	b	L8
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L31
	b	L8
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
	b	L8
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L125
L33:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L35:
L125:
	nop
L8:
	b	L124
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L126
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L126
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L126
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L126
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L126
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L42
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L43
	b	L126
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L37
L44:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L46
L45:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L46:
	b	L37
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L48
	b	L37
L47:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L49
L48:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L49:
	b	L37
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L37
L50:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L37
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L37
L53:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L55
L54:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L55:
	b	L37
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L37
L56:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L58
L57:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L58:
	b	L37
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L37
L59:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L61:
	b	L37
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L127
L62:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L64:
L127:
	nop
L37:
	b	L126
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L72
	b	L128
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L74
	b	L66
L73:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L75
L74:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L75:
	b	L66
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L77
	b	L66
L76:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L78
L77:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L78:
	b	L66
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L80
	b	L66
L79:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L81
L80:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L81:
	b	L66
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L83
	b	L66
L82:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L84
L83:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L84:
	b	L66
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L86
	b	L66
L85:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L87
L86:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L87:
	b	L66
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L89
	b	L66
L88:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L90
L89:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L90:
	b	L66
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L92
	b	L129
L91:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L93
L92:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L93:
L129:
	nop
L66:
	b	L128
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L94
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L97
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L98
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L99
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L100
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L101
	b	L130
L100:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L103
	b	L95
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
	b	L95
L101:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L95
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
	b	L95
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L109
	b	L95
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
	b	L95
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L111
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L112
	b	L95
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
	b	L95
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L115
	b	L95
L114:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L116
L115:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L116:
	b	L95
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L117
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L118
	b	L95
L117:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L119
L118:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L119:
	b	L95
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L120
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L121
	b	L131
L120:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L122
L121:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L122:
L131:
	nop
L95:
	b	L130
L124:
	nop
	b	L3
L126:
	nop
	b	L3
L128:
	nop
	b	L3
L130:
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
