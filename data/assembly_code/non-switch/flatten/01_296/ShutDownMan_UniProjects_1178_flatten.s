	.file	"ShutDownMan_UniProjects_1178_flatten.c"
	.text
	.globl	_TIG_IZ_g2ck_argc
	.bss
	.align 4
	.type	_TIG_IZ_g2ck_argc, @object
	.size	_TIG_IZ_g2ck_argc, 4
_TIG_IZ_g2ck_argc:
	.zero	4
	.globl	_TIG_IZ_g2ck_argv
	.align 8
	.type	_TIG_IZ_g2ck_argv, @object
	.size	_TIG_IZ_g2ck_argv, 8
_TIG_IZ_g2ck_argv:
	.zero	8
	.globl	_TIG_IZ_g2ck_envp
	.align 8
	.type	_TIG_IZ_g2ck_envp, @object
	.size	_TIG_IZ_g2ck_envp, 8
_TIG_IZ_g2ck_envp:
	.zero	8
	.text
	.globl	makeVet
	.type	makeVet, @function
makeVet:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movsd	%xmm0, -32(%rbp)
	movq	$0, -8(%rbp)
.L13:
	cmpq	$7, -8(%rbp)
	ja	.L14
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
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L14-.L4
	.long	.L14-.L4
	.long	.L14-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L15-.L4
	.text
.L7:
	movq	-24(%rbp), %rax
	movsd	-32(%rbp), %xmm0
	movsd	%xmm0, (%rax)
	movl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L9
.L5:
	cmpl	$99, -12(%rbp)
	jg	.L10
	movq	$5, -8(%rbp)
	jmp	.L9
.L10:
	movq	$7, -8(%rbp)
	jmp	.L9
.L6:
	movl	-12(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movsd	(%rax), %xmm0
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movsd	.LC0(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, (%rax)
	addl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L9
.L8:
	movq	$1, -8(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L13
.L15:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	makeVet, .-makeVet
	.section	.rodata
.LC1:
	.string	"N[%d] = %.4lf\n"
	.text
	.globl	printVet
	.type	printVet, @function
printVet:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L25:
	cmpq	$5, -8(%rbp)
	je	.L17
	cmpq	$5, -8(%rbp)
	ja	.L26
	cmpq	$2, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	ja	.L26
	cmpq	$0, -8(%rbp)
	je	.L27
	cmpq	$1, -8(%rbp)
	jne	.L26
	cmpl	$99, -12(%rbp)
	jg	.L21
	movq	$5, -8(%rbp)
	jmp	.L23
.L21:
	movq	$0, -8(%rbp)
	jmp	.L23
.L17:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L23
.L19:
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L23
.L26:
	nop
.L23:
	jmp	.L25
.L27:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	printVet, .-printVet
	.section	.rodata
.LC2:
	.string	"%lf"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$864, %rsp
	movl	%edi, -836(%rbp)
	movq	%rsi, -848(%rbp)
	movq	%rdx, -856(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_g2ck_envp(%rip)
	nop
.L29:
	movq	$0, _TIG_IZ_g2ck_argv(%rip)
	nop
.L30:
	movl	$0, _TIG_IZ_g2ck_argc(%rip)
	nop
	nop
.L31:
.L32:
#APP
# 64 "ShutDownMan_UniProjects_1178.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-g2ck--0
# 0 "" 2
#NO_APP
	movl	-836(%rbp), %eax
	movl	%eax, _TIG_IZ_g2ck_argc(%rip)
	movq	-848(%rbp), %rax
	movq	%rax, _TIG_IZ_g2ck_argv(%rip)
	movq	-856(%rbp), %rax
	movq	%rax, _TIG_IZ_g2ck_envp(%rip)
	nop
	movq	$0, -824(%rbp)
.L38:
	cmpq	$2, -824(%rbp)
	je	.L33
	cmpq	$2, -824(%rbp)
	ja	.L41
	cmpq	$0, -824(%rbp)
	je	.L35
	cmpq	$1, -824(%rbp)
	jne	.L41
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L39
	jmp	.L40
.L35:
	movq	$2, -824(%rbp)
	jmp	.L37
.L33:
	leaq	-832(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-832(%rbp), %rdx
	leaq	-816(%rbp), %rax
	movq	%rdx, %xmm0
	movq	%rax, %rdi
	call	makeVet
	leaq	-816(%rbp), %rax
	movq	%rax, %rdi
	call	printVet
	movq	$1, -824(%rbp)
	jmp	.L37
.L41:
	nop
.L37:
	jmp	.L38
.L40:
	call	__stack_chk_fail@PLT
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	0
	.long	1073741824
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
