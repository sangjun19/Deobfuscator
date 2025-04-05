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
	.ascii "1-1-4\0"
	.align	3
lC4:
	.ascii "1-2-1\0"
	.align	3
lC5:
	.ascii "1-2-2\0"
	.align	3
lC6:
	.ascii "1-2-3\0"
	.align	3
lC7:
	.ascii "1-2-4\0"
	.align	3
lC8:
	.ascii "2-1-1\0"
	.align	3
lC9:
	.ascii "2-1-2\0"
	.align	3
lC10:
	.ascii "2-1-3\0"
	.align	3
lC11:
	.ascii "2-1-4\0"
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
	.ascii "2-2-4\0"
	.align	3
lC16:
	.ascii "3-1-1\0"
	.align	3
lC17:
	.ascii "3-1-2\0"
	.align	3
lC18:
	.ascii "3-1-3\0"
	.align	3
lC19:
	.ascii "3-1-4\0"
	.align	3
lC20:
	.ascii "3-2-1\0"
	.align	3
lC21:
	.ascii "3-2-2\0"
	.align	3
lC22:
	.ascii "3-2-3\0"
	.align	3
lC23:
	.ascii "3-2-4\0"
	.align	3
lC24:
	.ascii "4-1-1\0"
	.align	3
lC25:
	.ascii "4-1-2\0"
	.align	3
lC26:
	.ascii "4-1-3\0"
	.align	3
lC27:
	.ascii "4-1-4\0"
	.align	3
lC28:
	.ascii "4-2-1\0"
	.align	3
lC29:
	.ascii "4-2-2\0"
	.align	3
lC30:
	.ascii "4-2-3\0"
	.align	3
lC31:
	.ascii "4-2-4\0"
	.align	3
lC32:
	.ascii "5-1-1\0"
	.align	3
lC33:
	.ascii "5-1-2\0"
	.align	3
lC34:
	.ascii "5-1-3\0"
	.align	3
lC35:
	.ascii "5-1-4\0"
	.align	3
lC36:
	.ascii "5-2-1\0"
	.align	3
lC37:
	.ascii "5-2-2\0"
	.align	3
lC38:
	.ascii "5-2-3\0"
	.align	3
lC39:
	.ascii "5-2-4\0"
	.align	3
lC40:
	.ascii "6-1-1\0"
	.align	3
lC41:
	.ascii "6-1-2\0"
	.align	3
lC42:
	.ascii "6-1-3\0"
	.align	3
lC43:
	.ascii "6-1-4\0"
	.align	3
lC44:
	.ascii "6-2-1\0"
	.align	3
lC45:
	.ascii "6-2-2\0"
	.align	3
lC46:
	.ascii "6-2-3\0"
	.align	3
lC47:
	.ascii "6-2-4\0"
	.align	3
lC48:
	.ascii "7-1-1\0"
	.align	3
lC49:
	.ascii "7-1-2\0"
	.align	3
lC50:
	.ascii "7-1-3\0"
	.align	3
lC51:
	.ascii "7-1-4\0"
	.align	3
lC52:
	.ascii "7-2-1\0"
	.align	3
lC53:
	.ascii "7-2-2\0"
	.align	3
lC54:
	.ascii "7-2-3\0"
	.align	3
lC55:
	.ascii "7-2-4\0"
	.align	3
lC56:
	.ascii "8-1-1\0"
	.align	3
lC57:
	.ascii "8-1-2\0"
	.align	3
lC58:
	.ascii "8-1-3\0"
	.align	3
lC59:
	.ascii "8-1-4\0"
	.align	3
lC60:
	.ascii "8-2-1\0"
	.align	3
lC61:
	.ascii "8-2-2\0"
	.align	3
lC62:
	.ascii "8-2-3\0"
	.align	3
lC63:
	.ascii "8-2-4\0"
	.align	3
lC64:
	.ascii "9-1-1\0"
	.align	3
lC65:
	.ascii "9-1-2\0"
	.align	3
lC66:
	.ascii "9-1-3\0"
	.align	3
lC67:
	.ascii "9-1-4\0"
	.align	3
lC68:
	.ascii "9-2-1\0"
	.align	3
lC69:
	.ascii "9-2-2\0"
	.align	3
lC70:
	.ascii "9-2-3\0"
	.align	3
lC71:
	.ascii "9-2-4\0"
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
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 4
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
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L13
	b	L3
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L19
	b	L130
L18:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L16
L19:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L16
L17:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L16
L15:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L16:
	b	L130
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L131
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L131
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L131
L23:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L21
L24:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L21
L22:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L21
L20:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L21:
	b	L131
L130:
	nop
	b	L3
L131:
	nop
	b	L3
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L26
	b	L3
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L132
L31:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L29
L32:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L29
L30:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L29
L28:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L29:
	b	L132
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L133
L36:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L34
L37:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L34
L35:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L34
L33:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L34:
	b	L133
L132:
	nop
	b	L3
L133:
	nop
	b	L3
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L39
	b	L3
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L134
L44:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L42
L45:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L42
L43:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L42:
	b	L134
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L135
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L135
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L135
L49:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L47
L50:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L47
L48:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L47:
	b	L135
L134:
	nop
	b	L3
L135:
	nop
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L52
	b	L3
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L54
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L136
L57:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L55
L58:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L55
L56:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L55
L54:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L55:
	b	L136
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L137
L62:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L60
L63:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L60
L61:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L60
L59:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L60:
	b	L137
L136:
	nop
	b	L3
L137:
	nop
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L65
	b	L3
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L67
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L71
	b	L138
L70:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L68
L71:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L68
L69:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L68
L67:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L68:
	b	L138
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L72
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L139
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L139
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L139
L75:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L73
L76:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L73
L74:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L73
L72:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L73:
	b	L139
L138:
	nop
	b	L3
L139:
	nop
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L78
	b	L3
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L140
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L84
	b	L140
L83:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L81
L84:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L81
L82:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L81
L80:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L81:
	b	L140
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L141
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L141
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L89
	b	L141
L88:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L86
L89:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L86
L87:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L86
L85:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L86:
	b	L141
L140:
	nop
	b	L3
L141:
	nop
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L91
	b	L3
L90:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L142
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L142
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L96
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L97
	b	L142
L96:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L94
L97:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L94
L95:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L94
L93:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L94:
	b	L142
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L98
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L143
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L143
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L143
L101:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L99
L102:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	b	L99
L100:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L99
L98:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L99:
	b	L143
L142:
	nop
	b	L3
L143:
	nop
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L103
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L104
	b	L3
L103:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L106
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L144
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L110
	b	L144
L109:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L107
L110:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L107
L108:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L107
L106:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L107:
	b	L144
L104:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L111
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L113
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L115
	b	L145
L114:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L112
L115:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	b	L112
L113:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L112
L111:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L112:
	b	L145
L144:
	nop
	b	L3
L145:
	nop
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L116
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L117
	b	L148
L116:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L119
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L121
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L122
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L123
	b	L146
L122:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L120
L123:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	b	L120
L121:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L120
L119:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L120:
	b	L146
L117:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L124
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L126
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L127
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L128
	b	L147
L127:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L125
L128:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	b	L125
L126:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L125
L124:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L125:
	b	L147
L146:
	nop
	b	L148
L147:
	nop
L148:
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
