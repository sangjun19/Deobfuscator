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
	.align	3
lC60:
	.ascii "6-1-1\0"
	.align	3
lC61:
	.ascii "6-1-2\0"
	.align	3
lC62:
	.ascii "6-2-1\0"
	.align	3
lC63:
	.ascii "6-2-2\0"
	.align	3
lC64:
	.ascii "6-3-1\0"
	.align	3
lC65:
	.ascii "6-3-2\0"
	.align	3
lC66:
	.ascii "6-4-1\0"
	.align	3
lC67:
	.ascii "6-4-2\0"
	.align	3
lC68:
	.ascii "6-5-1\0"
	.align	3
lC69:
	.ascii "6-5-2\0"
	.align	3
lC70:
	.ascii "6-6-1\0"
	.align	3
lC71:
	.ascii "6-6-2\0"
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
	mov	w0, 6
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
	cmp	w0, 6
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L160
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L160
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L10
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
	b	L10
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L10
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
	b	L10
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L10
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
	b	L10
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L10
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
	b	L10
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L10
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
	b	L10
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L161
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
L161:
	nop
L10:
	b	L160
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L162
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L40
	b	L162
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L35
L41:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L43:
	b	L35
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L35
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
	b	L35
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L48
	b	L35
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
	b	L35
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L35
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
	b	L35
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L35
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
	b	L35
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L163
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
L163:
	nop
L35:
	b	L162
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L59
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L61
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L62
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L63
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L65
	b	L164
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L66
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L67
	b	L60
L66:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L68
L67:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L68:
	b	L60
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L70
	b	L60
L69:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L71
L70:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L71:
	b	L60
L63:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L72
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L73
	b	L60
L72:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L74
L73:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L74:
	b	L60
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L60
L75:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L77
L76:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L77:
	b	L60
L61:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L78
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L79
	b	L60
L78:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L80
L79:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L80:
	b	L60
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L82
	b	L165
L81:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L83
L82:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L83:
L165:
	nop
L60:
	b	L164
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L84
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L86
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L87
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L166
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L90
	b	L166
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L92
	b	L85
L91:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L93
L92:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L93:
	b	L85
L90:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L94
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L95
	b	L85
L94:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L96
L95:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L96:
	b	L85
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L85
L97:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L99
L98:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L99:
	b	L85
L87:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L101
	b	L85
L100:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L102
L101:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L102:
	b	L85
L86:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L103
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L104
	b	L85
L103:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L105
L104:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L105:
	b	L85
L84:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L106
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L107
	b	L167
L106:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L108
L107:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L108:
L167:
	nop
L85:
	b	L166
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L109
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L111
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L112
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L113
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L168
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L114
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L115
	b	L168
L114:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L116
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L117
	b	L110
L116:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L118
L117:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L118:
	b	L110
L115:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L119
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L120
	b	L110
L119:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L121
L120:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L121:
	b	L110
L113:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L122
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L123
	b	L110
L122:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L124
L123:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L124:
	b	L110
L112:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L125
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L126
	b	L110
L125:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L127
L126:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L127:
	b	L110
L111:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L128
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L129
	b	L110
L128:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L130
L129:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L130:
	b	L110
L109:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L131
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L132
	b	L169
L131:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L133
L132:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L133:
L169:
	nop
L110:
	b	L168
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L134
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L136
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L137
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L138
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L170
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L139
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L140
	b	L170
L139:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L141
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L142
	b	L135
L141:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L143
L142:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
L143:
	b	L135
L140:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L144
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L145
	b	L135
L144:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L146
L145:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L146:
	b	L135
L138:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L147
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L148
	b	L135
L147:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L149
L148:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	nop
L149:
	b	L135
L137:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L150
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L151
	b	L135
L150:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L152
L151:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L152:
	b	L135
L136:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L153
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L154
	b	L135
L153:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L155
L154:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	nop
L155:
	b	L135
L134:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L156
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L157
	b	L171
L156:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L158
L157:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L158:
L171:
	nop
L135:
	b	L170
L160:
	nop
	b	L3
L162:
	nop
	b	L3
L164:
	nop
	b	L3
L166:
	nop
	b	L3
L168:
	nop
	b	L3
L170:
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
