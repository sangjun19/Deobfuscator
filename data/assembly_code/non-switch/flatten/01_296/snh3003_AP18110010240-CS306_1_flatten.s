	.file	"snh3003_AP18110010240-CS306_1_flatten.c"
	.text
	.globl	string
	.bss
	.align 32
	.type	string, @object
	.size	string, 50
string:
	.zero	50
	.globl	_TIG_IZ_oYSI_envp
	.align 8
	.type	_TIG_IZ_oYSI_envp, @object
	.size	_TIG_IZ_oYSI_envp, 8
_TIG_IZ_oYSI_envp:
	.zero	8
	.globl	_TIG_IZ_oYSI_argv
	.align 8
	.type	_TIG_IZ_oYSI_argv, @object
	.size	_TIG_IZ_oYSI_argv, 8
_TIG_IZ_oYSI_argv:
	.zero	8
	.globl	_TIG_IZ_oYSI_argc
	.align 4
	.type	_TIG_IZ_oYSI_argc, @object
	.size	_TIG_IZ_oYSI_argc, 4
_TIG_IZ_oYSI_argc:
	.zero	4
	.globl	ip
	.align 8
	.type	ip, @object
	.size	ip, 8
ip:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%s\tE'->+TE' \n"
.LC1:
	.string	"%s\tE'->^ \n"
	.text
	.globl	Edash
	.type	Edash, @function
Edash:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L22:
	cmpq	$11, -8(%rbp)
	ja	.L23
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L23-.L4
	.long	.L23-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	movq	ip(%rip), %rax
	movzbl	(%rax), %eax
	cmpb	$43, %al
	jne	.L14
	movq	$0, -8(%rbp)
	jmp	.L16
.L14:
	movq	$2, -8(%rbp)
	jmp	.L16
.L7:
	movl	$1, %eax
	jmp	.L17
.L12:
	movl	$1, %eax
	jmp	.L17
.L10:
	cmpl	$0, -12(%rbp)
	je	.L18
	movq	$5, -8(%rbp)
	jmp	.L16
.L18:
	movq	$10, -8(%rbp)
	jmp	.L16
.L3:
	cmpl	$0, -16(%rbp)
	je	.L20
	movq	$8, -8(%rbp)
	jmp	.L16
.L20:
	movq	$9, -8(%rbp)
	jmp	.L16
.L6:
	movl	$0, %eax
	jmp	.L17
.L8:
	call	Edash
	movl	%eax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L16
.L5:
	movl	$0, %eax
	jmp	.L17
.L13:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ip(%rip), %rax
	addq	$1, %rax
	movq	%rax, ip(%rip)
	call	T
	movl	%eax, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L16
.L11:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L23:
	nop
.L16:
	jmp	.L22
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	Edash, .-Edash
	.section	.rodata
.LC2:
	.string	"%s\tF->(E) \n"
.LC3:
	.string	"%s\tF->id \n"
	.text
	.globl	F
	.type	F, @function
F:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L49:
	cmpq	$13, -8(%rbp)
	ja	.L50
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L27(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L27(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L27:
	.long	.L38-.L27
	.long	.L37-.L27
	.long	.L36-.L27
	.long	.L50-.L27
	.long	.L35-.L27
	.long	.L34-.L27
	.long	.L33-.L27
	.long	.L50-.L27
	.long	.L32-.L27
	.long	.L31-.L27
	.long	.L30-.L27
	.long	.L29-.L27
	.long	.L28-.L27
	.long	.L26-.L27
	.text
.L35:
	cmpl	$0, -12(%rbp)
	je	.L39
	movq	$9, -8(%rbp)
	jmp	.L41
.L39:
	movq	$6, -8(%rbp)
	jmp	.L41
.L28:
	movq	ip(%rip), %rax
	movzbl	(%rax), %eax
	cmpb	$105, %al
	jne	.L42
	movq	$2, -8(%rbp)
	jmp	.L41
.L42:
	movq	$0, -8(%rbp)
	jmp	.L41
.L32:
	movq	ip(%rip), %rax
	addq	$1, %rax
	movq	%rax, ip(%rip)
	movq	$11, -8(%rbp)
	jmp	.L41
.L37:
	movq	ip(%rip), %rax
	movzbl	(%rax), %eax
	cmpb	$40, %al
	jne	.L44
	movq	$10, -8(%rbp)
	jmp	.L41
.L44:
	movq	$12, -8(%rbp)
	jmp	.L41
.L29:
	movl	$1, %eax
	jmp	.L46
.L31:
	movq	ip(%rip), %rax
	movzbl	(%rax), %eax
	cmpb	$41, %al
	jne	.L47
	movq	$8, -8(%rbp)
	jmp	.L41
.L47:
	movq	$5, -8(%rbp)
	jmp	.L41
.L26:
	movl	$1, %eax
	jmp	.L46
.L33:
	movl	$0, %eax
	jmp	.L46
.L34:
	movl	$0, %eax
	jmp	.L46
.L30:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ip(%rip), %rax
	addq	$1, %rax
	movq	%rax, ip(%rip)
	call	E
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L41
.L38:
	movl	$0, %eax
	jmp	.L46
.L36:
	movq	ip(%rip), %rax
	addq	$1, %rax
	movq	%rax, ip(%rip)
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -8(%rbp)
	jmp	.L41
.L50:
	nop
.L41:
	jmp	.L49
.L46:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	F, .-F
	.section	.rodata
	.align 8
.LC4:
	.string	"\n--------------------------------"
.LC5:
	.string	"Error in parsing String"
	.align 8
.LC6:
	.string	"\n String is successfully parsed"
.LC7:
	.string	"Enter the string:"
.LC8:
	.string	"%s"
	.align 8
.LC9:
	.string	"\n\nInput\tAction\n--------------------------------"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, -16(%rbp)
	jmp	.L52
.L53:
	movl	-16(%rbp), %eax
	cltq
	leaq	string(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -16(%rbp)
.L52:
	cmpl	$49, -16(%rbp)
	jle	.L53
	nop
.L54:
	movq	$0, ip(%rip)
	nop
.L55:
	movq	$0, _TIG_IZ_oYSI_envp(%rip)
	nop
.L56:
	movq	$0, _TIG_IZ_oYSI_argv(%rip)
	nop
.L57:
	movl	$0, _TIG_IZ_oYSI_argc(%rip)
	nop
	nop
.L58:
.L59:
#APP
# 119 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-oYSI--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_oYSI_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_oYSI_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_oYSI_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L76:
	cmpq	$10, -8(%rbp)
	ja	.L78
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L62(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L62(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L62:
	.long	.L69-.L62
	.long	.L78-.L62
	.long	.L78-.L62
	.long	.L68-.L62
	.long	.L78-.L62
	.long	.L67-.L62
	.long	.L66-.L62
	.long	.L65-.L62
	.long	.L64-.L62
	.long	.L63-.L62
	.long	.L61-.L62
	.text
.L64:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L70
.L68:
	movl	$0, %eax
	jmp	.L77
.L63:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L70
.L66:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L70
.L67:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	string(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	string(%rip), %rax
	movq	%rax, ip(%rip)
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	E
	movl	%eax, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L70
.L61:
	cmpl	$0, -12(%rbp)
	je	.L72
	movq	$7, -8(%rbp)
	jmp	.L70
.L72:
	movq	$8, -8(%rbp)
	jmp	.L70
.L69:
	movq	$5, -8(%rbp)
	jmp	.L70
.L65:
	movq	ip(%rip), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L74
	movq	$9, -8(%rbp)
	jmp	.L70
.L74:
	movq	$6, -8(%rbp)
	jmp	.L70
.L78:
	nop
.L70:
	jmp	.L76
.L77:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"%s\tT->FT' \n"
	.text
	.globl	T
	.type	T, @function
T:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$5, -8(%rbp)
.L96:
	cmpq	$8, -8(%rbp)
	ja	.L97
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L82(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L82(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L82:
	.long	.L89-.L82
	.long	.L88-.L82
	.long	.L87-.L82
	.long	.L97-.L82
	.long	.L86-.L82
	.long	.L85-.L82
	.long	.L84-.L82
	.long	.L83-.L82
	.long	.L81-.L82
	.text
.L86:
	movl	$0, %eax
	jmp	.L90
.L81:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	F
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L91
.L88:
	cmpl	$0, -12(%rbp)
	je	.L92
	movq	$7, -8(%rbp)
	jmp	.L91
.L92:
	movq	$2, -8(%rbp)
	jmp	.L91
.L84:
	cmpl	$0, -16(%rbp)
	je	.L94
	movq	$0, -8(%rbp)
	jmp	.L91
.L94:
	movq	$4, -8(%rbp)
	jmp	.L91
.L85:
	movq	$8, -8(%rbp)
	jmp	.L91
.L89:
	movl	$1, %eax
	jmp	.L90
.L83:
	call	Tdash
	movl	%eax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L91
.L87:
	movl	$0, %eax
	jmp	.L90
.L97:
	nop
.L91:
	jmp	.L96
.L90:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	T, .-T
	.section	.rodata
.LC11:
	.string	"%s\tT'->*FT' \n"
.LC12:
	.string	"%s\tT'->^ \n"
	.text
	.globl	Tdash
	.type	Tdash, @function
Tdash:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$7, -8(%rbp)
.L119:
	cmpq	$10, -8(%rbp)
	ja	.L120
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L101(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L101(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L101:
	.long	.L110-.L101
	.long	.L109-.L101
	.long	.L108-.L101
	.long	.L107-.L101
	.long	.L106-.L101
	.long	.L120-.L101
	.long	.L105-.L101
	.long	.L104-.L101
	.long	.L103-.L101
	.long	.L102-.L101
	.long	.L100-.L101
	.text
.L106:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ip(%rip), %rax
	addq	$1, %rax
	movq	%rax, ip(%rip)
	call	F
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L111
.L103:
	call	Tdash
	movl	%eax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L111
.L109:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L111
.L107:
	movl	$0, %eax
	jmp	.L112
.L102:
	movl	$1, %eax
	jmp	.L112
.L105:
	movl	$0, %eax
	jmp	.L112
.L100:
	movl	$1, %eax
	jmp	.L112
.L110:
	cmpl	$0, -12(%rbp)
	je	.L113
	movq	$8, -8(%rbp)
	jmp	.L111
.L113:
	movq	$6, -8(%rbp)
	jmp	.L111
.L104:
	movq	ip(%rip), %rax
	movzbl	(%rax), %eax
	cmpb	$42, %al
	jne	.L115
	movq	$4, -8(%rbp)
	jmp	.L111
.L115:
	movq	$1, -8(%rbp)
	jmp	.L111
.L108:
	cmpl	$0, -16(%rbp)
	je	.L117
	movq	$10, -8(%rbp)
	jmp	.L111
.L117:
	movq	$3, -8(%rbp)
	jmp	.L111
.L120:
	nop
.L111:
	jmp	.L119
.L112:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	Tdash, .-Tdash
	.section	.rodata
.LC13:
	.string	"%s\tE->TE' \n"
	.text
	.globl	E
	.type	E, @function
E:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$6, -8(%rbp)
.L138:
	cmpq	$8, -8(%rbp)
	ja	.L139
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L124(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L124(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L124:
	.long	.L139-.L124
	.long	.L131-.L124
	.long	.L130-.L124
	.long	.L129-.L124
	.long	.L128-.L124
	.long	.L127-.L124
	.long	.L126-.L124
	.long	.L125-.L124
	.long	.L123-.L124
	.text
.L128:
	call	Edash
	movl	%eax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L132
.L123:
	movq	ip(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	T
	movl	%eax, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L132
.L131:
	movl	$0, %eax
	jmp	.L133
.L129:
	movl	$1, %eax
	jmp	.L133
.L126:
	movq	$8, -8(%rbp)
	jmp	.L132
.L127:
	cmpl	$0, -16(%rbp)
	je	.L134
	movq	$3, -8(%rbp)
	jmp	.L132
.L134:
	movq	$2, -8(%rbp)
	jmp	.L132
.L125:
	cmpl	$0, -12(%rbp)
	je	.L136
	movq	$4, -8(%rbp)
	jmp	.L132
.L136:
	movq	$1, -8(%rbp)
	jmp	.L132
.L130:
	movl	$0, %eax
	jmp	.L133
.L139:
	nop
.L132:
	jmp	.L138
.L133:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	E, .-E
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
