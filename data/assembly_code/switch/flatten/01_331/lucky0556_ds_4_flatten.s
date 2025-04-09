	.file	"lucky0556_ds_4_flatten.c"
	.text
	.globl	i
	.bss
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.zero	4
	.globl	_TIG_IZ_VHlT_argc
	.align 4
	.type	_TIG_IZ_VHlT_argc, @object
	.size	_TIG_IZ_VHlT_argc, 4
_TIG_IZ_VHlT_argc:
	.zero	4
	.globl	s
	.align 32
	.type	s, @object
	.size	s, 40
s:
	.zero	40
	.globl	ele
	.align 4
	.type	ele, @object
	.size	ele, 4
ele:
	.zero	4
	.globl	_TIG_IZ_VHlT_argv
	.align 8
	.type	_TIG_IZ_VHlT_argv, @object
	.size	_TIG_IZ_VHlT_argv, 8
_TIG_IZ_VHlT_argv:
	.zero	8
	.globl	op1
	.align 4
	.type	op1, @object
	.size	op1, 4
op1:
	.zero	4
	.globl	op2
	.align 4
	.type	op2, @object
	.size	op2, 4
op2:
	.zero	4
	.globl	n
	.align 4
	.type	n, @object
	.size	n, 4
n:
	.zero	4
	.globl	res
	.align 4
	.type	res, @object
	.size	res, 4
res:
	.zero	4
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.globl	_TIG_IZ_VHlT_envp
	.align 8
	.type	_TIG_IZ_VHlT_envp, @object
	.size	_TIG_IZ_VHlT_envp, 8
_TIG_IZ_VHlT_envp:
	.zero	8
	.text
	.globl	push
	.type	push, @function
push:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movq	$0, -8(%rbp)
	jmp	.L5
.L4:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	s(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$2, -8(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	push, .-push
	.globl	pop
	.type	pop, @function
pop:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L16:
	cmpq	$2, -8(%rbp)
	je	.L11
	cmpq	$2, -8(%rbp)
	ja	.L18
	cmpq	$0, -8(%rbp)
	je	.L13
	cmpq	$1, -8(%rbp)
	jne	.L18
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$2, -8(%rbp)
	jmp	.L14
.L13:
	movq	$1, -8(%rbp)
	jmp	.L14
.L11:
	movl	-12(%rbp), %eax
	jmp	.L17
.L18:
	nop
.L14:
	jmp	.L16
.L17:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	pop, .-pop
	.section	.rodata
	.align 8
.LC0:
	.string	"\n1.Tower of Hanoi\n2.Postfix Evaluation"
.LC1:
	.string	"enter your choice"
.LC2:
	.string	"%d"
.LC3:
	.string	"enter the number of disks"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, n(%rip)
	nop
.L20:
	movl	$0, ele(%rip)
	nop
.L21:
	movl	$0, -20(%rbp)
	jmp	.L22
.L23:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L22:
	cmpl	$9, -20(%rbp)
	jle	.L23
	nop
.L24:
	movl	$-1, top(%rip)
	nop
.L25:
	movl	$0, i(%rip)
	nop
.L26:
	movl	$0, res(%rip)
	nop
.L27:
	movl	$0, op2(%rip)
	nop
.L28:
	movl	$0, op1(%rip)
	nop
.L29:
	movq	$0, _TIG_IZ_VHlT_envp(%rip)
	nop
.L30:
	movq	$0, _TIG_IZ_VHlT_argv(%rip)
	nop
.L31:
	movl	$0, _TIG_IZ_VHlT_argc(%rip)
	nop
	nop
.L32:
.L33:
#APP
# 133 "lucky0556_ds_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-VHlT--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_VHlT_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_VHlT_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_VHlT_envp(%rip)
	nop
	movq	$11, -16(%rbp)
.L51:
	cmpq	$11, -16(%rbp)
	ja	.L54
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L36(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L36(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L36:
	.long	.L43-.L36
	.long	.L54-.L36
	.long	.L42-.L36
	.long	.L54-.L36
	.long	.L41-.L36
	.long	.L54-.L36
	.long	.L55-.L36
	.long	.L39-.L36
	.long	.L38-.L36
	.long	.L54-.L36
	.long	.L37-.L36
	.long	.L35-.L36
	.text
.L41:
	call	eval
	movq	$8, -16(%rbp)
	jmp	.L44
.L38:
	movl	-24(%rbp), %eax
	cmpl	$2, %eax
	jg	.L45
	movq	$10, -16(%rbp)
	jmp	.L44
.L45:
	movq	$6, -16(%rbp)
	jmp	.L44
.L35:
	movq	$10, -16(%rbp)
	jmp	.L44
.L37:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L44
.L43:
	movq	$8, -16(%rbp)
	jmp	.L44
.L39:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	n(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	n(%rip), %eax
	movl	$66, %ecx
	movl	$67, %edx
	movl	$65, %esi
	movl	%eax, %edi
	call	tow
	movq	$8, -16(%rbp)
	jmp	.L44
.L42:
	movl	-24(%rbp), %eax
	cmpl	$1, %eax
	je	.L48
	cmpl	$2, %eax
	jne	.L49
	movq	$4, -16(%rbp)
	jmp	.L50
.L48:
	movq	$7, -16(%rbp)
	jmp	.L50
.L49:
	movq	$0, -16(%rbp)
	nop
.L50:
	jmp	.L44
.L54:
	nop
.L44:
	jmp	.L51
.L55:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L53
	call	__stack_chk_fail@PLT
.L53:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC4:
	.string	"\nMove disk 1 from rod %c to rod %c"
	.align 8
.LC5:
	.string	"\nMove disk %d from rod %c to rod %c"
	.text
	.globl	tow
	.type	tow, @function
tow:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%ecx, %eax
	movl	%esi, %ecx
	movb	%cl, -24(%rbp)
	movb	%dl, -28(%rbp)
	movb	%al, -32(%rbp)
	movq	$3, -8(%rbp)
.L68:
	cmpq	$5, -8(%rbp)
	ja	.L69
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L59(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L59(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L59:
	.long	.L69-.L59
	.long	.L70-.L59
	.long	.L62-.L59
	.long	.L61-.L59
	.long	.L70-.L59
	.long	.L58-.L59
	.text
.L61:
	cmpl	$1, -20(%rbp)
	jne	.L65
	movq	$5, -8(%rbp)
	jmp	.L67
.L65:
	movq	$2, -8(%rbp)
	jmp	.L67
.L58:
	movsbl	-32(%rbp), %edx
	movsbl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L67
.L62:
	movsbl	-28(%rbp), %ecx
	movsbl	-32(%rbp), %edx
	movsbl	-24(%rbp), %eax
	movl	-20(%rbp), %esi
	leal	-1(%rsi), %edi
	movl	%eax, %esi
	call	tow
	movsbl	-32(%rbp), %ecx
	movsbl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movsbl	-32(%rbp), %ecx
	movsbl	-24(%rbp), %edx
	movsbl	-28(%rbp), %eax
	movl	-20(%rbp), %esi
	leal	-1(%rsi), %edi
	movl	%eax, %esi
	call	tow
	movq	$1, -8(%rbp)
	jmp	.L67
.L69:
	nop
.L67:
	jmp	.L68
.L70:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	tow, .-tow
	.section	.rodata
.LC6:
	.string	"result of the postfix exp %d\n"
.LC7:
	.string	"enter the postfix"
.LC8:
	.string	"%s"
	.text
	.globl	eval
	.type	eval, @function
eval:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$16, -40(%rbp)
.L103:
	cmpq	$24, -40(%rbp)
	ja	.L106
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L74(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L74(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L74:
	.long	.L90-.L74
	.long	.L106-.L74
	.long	.L89-.L74
	.long	.L88-.L74
	.long	.L106-.L74
	.long	.L107-.L74
	.long	.L106-.L74
	.long	.L86-.L74
	.long	.L106-.L74
	.long	.L106-.L74
	.long	.L85-.L74
	.long	.L84-.L74
	.long	.L83-.L74
	.long	.L82-.L74
	.long	.L81-.L74
	.long	.L80-.L74
	.long	.L79-.L74
	.long	.L78-.L74
	.long	.L106-.L74
	.long	.L77-.L74
	.long	.L106-.L74
	.long	.L106-.L74
	.long	.L76-.L74
	.long	.L75-.L74
	.long	.L73-.L74
	.text
.L81:
	movl	res(%rip), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -40(%rbp)
	jmp	.L91
.L80:
	movl	op1(%rip), %edx
	movl	op2(%rip), %eax
	addl	%edx, %eax
	movl	%eax, res(%rip)
	movq	$10, -40(%rbp)
	jmp	.L91
.L83:
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$23, -40(%rbp)
	jmp	.L91
.L75:
	movl	i(%rip), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	testb	%al, %al
	je	.L92
	movq	$22, -40(%rbp)
	jmp	.L91
.L92:
	movq	$14, -40(%rbp)
	jmp	.L91
.L88:
	movl	op1(%rip), %eax
	movl	op2(%rip), %ecx
	cltd
	idivl	%ecx
	movl	%eax, res(%rip)
	movq	$10, -40(%rbp)
	jmp	.L91
.L79:
	movq	$11, -40(%rbp)
	jmp	.L91
.L73:
	movsbl	-49(%rbp), %eax
	subl	$48, %eax
	movl	%eax, %edi
	call	push
	movq	$12, -40(%rbp)
	jmp	.L91
.L84:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, i(%rip)
	movq	$23, -40(%rbp)
	jmp	.L91
.L82:
	movl	op1(%rip), %edx
	movl	op2(%rip), %eax
	imull	%edx, %eax
	movl	%eax, res(%rip)
	movq	$10, -40(%rbp)
	jmp	.L91
.L77:
	movq	$10, -40(%rbp)
	jmp	.L91
.L78:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-49(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L94
	movq	$24, -40(%rbp)
	jmp	.L91
.L94:
	movq	$7, -40(%rbp)
	jmp	.L91
.L76:
	movl	i(%rip), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movb	%al, -49(%rbp)
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$17, -40(%rbp)
	jmp	.L91
.L85:
	movl	res(%rip), %eax
	movl	%eax, %edi
	call	push
	movq	$12, -40(%rbp)
	jmp	.L91
.L90:
	movsbl	-49(%rbp), %eax
	cmpl	$47, %eax
	je	.L97
	cmpl	$47, %eax
	jg	.L98
	cmpl	$45, %eax
	je	.L99
	cmpl	$45, %eax
	jg	.L98
	cmpl	$42, %eax
	je	.L100
	cmpl	$43, %eax
	je	.L101
	jmp	.L98
.L97:
	movq	$3, -40(%rbp)
	jmp	.L102
.L100:
	movq	$13, -40(%rbp)
	jmp	.L102
.L99:
	movq	$2, -40(%rbp)
	jmp	.L102
.L101:
	movq	$15, -40(%rbp)
	jmp	.L102
.L98:
	movq	$19, -40(%rbp)
	nop
.L102:
	jmp	.L91
.L86:
	call	pop
	movl	%eax, op2(%rip)
	call	pop
	movl	%eax, op1(%rip)
	movq	$0, -40(%rbp)
	jmp	.L91
.L89:
	movl	op1(%rip), %eax
	movl	op2(%rip), %edx
	subl	%edx, %eax
	movl	%eax, res(%rip)
	movq	$10, -40(%rbp)
	jmp	.L91
.L106:
	nop
.L91:
	jmp	.L103
.L107:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L105
	call	__stack_chk_fail@PLT
.L105:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	eval, .-eval
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
