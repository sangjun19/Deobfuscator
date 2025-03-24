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
	.align	3
lC56:
	.ascii "5-1-1\0"
	.align	3
lC57:
	.ascii "5-1-2\0"
	.align	3
lC58:
	.ascii "5-2-1\0"
	.align	3
lC59:
	.ascii "5-2-2\0"
	.align	3
lC60:
	.ascii "5-3-1\0"
	.align	3
lC61:
	.ascii "5-3-2\0"
	.align	3
lC62:
	.ascii "5-4-1\0"
	.align	3
lC63:
	.ascii "5-4-2\0"
	.align	3
lC64:
	.ascii "5-5-1\0"
	.align	3
lC65:
	.ascii "5-5-2\0"
	.align	3
lC66:
	.ascii "5-6-1\0"
	.align	3
lC67:
	.ascii "5-6-2\0"
	.align	3
lC68:
	.ascii "5-7-1\0"
	.align	3
lC69:
	.ascii "5-7-2\0"
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
	mov	w0, 7
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
	cmp	w0, 7
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L154
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L154
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L9
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
	b	L9
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L9
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
	b	L9
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L9
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
	b	L9
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L9
L25:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L27:
	b	L9
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L9
L28:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L9
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L9
L31:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L33
L32:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L33:
	b	L9
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L35
	b	L155
L34:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L36
L35:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L36:
L155:
	nop
L9:
	b	L154
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L42
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L44
	b	L156
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L38
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
	b	L38
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L49
	b	L38
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
	b	L38
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L52
	b	L38
L51:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L53
L52:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L53:
	b	L38
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L55
	b	L38
L54:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L56
L55:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L56:
	b	L38
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L38
L57:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L59
L58:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L59:
	b	L38
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L61
	b	L38
L60:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L62
L61:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L62:
	b	L38
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L64
	b	L157
L63:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L65
L64:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L65:
L157:
	nop
L38:
	b	L156
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L71
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L158
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L72
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L73
	b	L158
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L75
	b	L67
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
	b	L67
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L78
	b	L67
L77:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L79
L78:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L79:
	b	L67
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L81
	b	L67
L80:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L67
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L84
	b	L67
L83:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L85
L84:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L85:
	b	L67
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L87
	b	L67
L86:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L88
L87:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L88:
	b	L67
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L90
	b	L67
L89:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L91
L90:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L91:
	b	L67
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L92
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L93
	b	L159
L92:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L94
L93:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L94:
L159:
	nop
L67:
	b	L158
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L95
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L97
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L98
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L99
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L100
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L102
	b	L160
L101:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L104
	b	L96
L103:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L105
L104:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L105:
	b	L96
L102:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L106
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L107
	b	L96
L106:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L108
L107:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L108:
	b	L96
L100:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L110
	b	L96
L109:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L111
L110:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L111:
	b	L96
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L112
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L113
	b	L96
L112:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L114
L113:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L114:
	b	L96
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L115
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L116
	b	L96
L115:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L117
L116:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L117:
	b	L96
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L118
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L119
	b	L96
L118:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L120
L119:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L120:
	b	L96
L95:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L122
	b	L161
L121:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L123
L122:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L123:
L161:
	nop
L96:
	b	L160
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L124
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L126
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L127
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L128
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L129
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L130
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L131
	b	L162
L130:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L132
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L133
	b	L125
L132:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L134
L133:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L134:
	b	L125
L131:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L135
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L136
	b	L125
L135:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L137
L136:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L137:
	b	L125
L129:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L138
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L139
	b	L125
L138:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L140
L139:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
L140:
	b	L125
L128:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L141
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L142
	b	L125
L141:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L143
L142:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L143:
	b	L125
L127:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L144
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L145
	b	L125
L144:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L146
L145:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	nop
L146:
	b	L125
L126:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L147
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L148
	b	L125
L147:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L149
L148:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L149:
	b	L125
L124:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L150
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L151
	b	L163
L150:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L152
L151:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	nop
L152:
L163:
	nop
L125:
	b	L162
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
