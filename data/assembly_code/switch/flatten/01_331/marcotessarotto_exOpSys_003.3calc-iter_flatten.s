	.file	"marcotessarotto_exOpSys_003.3calc-iter_flatten.c"
	.text
	.globl	_TIG_IZ_MtIt_envp
	.bss
	.align 8
	.type	_TIG_IZ_MtIt_envp, @object
	.size	_TIG_IZ_MtIt_envp, 8
_TIG_IZ_MtIt_envp:
	.zero	8
	.globl	_TIG_IZ_MtIt_argv
	.align 8
	.type	_TIG_IZ_MtIt_argv, @object
	.size	_TIG_IZ_MtIt_argv, 8
_TIG_IZ_MtIt_argv:
	.zero	8
	.globl	_TIG_IZ_MtIt_argc
	.align 4
	.type	_TIG_IZ_MtIt_argc, @object
	.size	_TIG_IZ_MtIt_argc, 4
_TIG_IZ_MtIt_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%c %d %d => %d\n"
.LC1:
	.string	"EOF!!!!"
.LC2:
	.string	"wrong parameters, control=%d\n"
.LC3:
	.string	"wrong operation\n"
.LC4:
	.string	" %c %d %d "
	.text
	.globl	calculator
	.type	calculator, @function
calculator:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -16(%rbp)
.L33:
	cmpq	$21, -16(%rbp)
	ja	.L36
	movq	-16(%rbp), %rax
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
	.long	.L36-.L4
	.long	.L36-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L36-.L4
	.long	.L15-.L4
	.long	.L36-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L36-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movl	-28(%rbp), %ecx
	movl	-32(%rbp), %edx
	movzbl	-33(%rbp), %eax
	movsbl	%al, %eax
	movl	-24(%rbp), %esi
	movl	%esi, %r8d
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$19, -16(%rbp)
	jmp	.L21
.L18:
	movq	$5, -16(%rbp)
	jmp	.L21
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L21
.L9:
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -24(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L21
.L12:
	movzbl	-33(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L22
	cmpl	$47, %eax
	jg	.L23
	cmpl	$45, %eax
	je	.L24
	cmpl	$45, %eax
	jg	.L23
	cmpl	$42, %eax
	je	.L25
	cmpl	$43, %eax
	je	.L26
	jmp	.L23
.L22:
	movq	$11, -16(%rbp)
	jmp	.L27
.L25:
	movq	$10, -16(%rbp)
	jmp	.L27
.L24:
	movq	$3, -16(%rbp)
	jmp	.L27
.L26:
	movq	$15, -16(%rbp)
	jmp	.L27
.L23:
	movq	$6, -16(%rbp)
	nop
.L27:
	jmp	.L21
.L15:
	cmpl	$-1, -20(%rbp)
	jne	.L28
	movq	$14, -16(%rbp)
	jmp	.L21
.L28:
	movq	$21, -16(%rbp)
	jmp	.L21
.L19:
	movl	-32(%rbp), %eax
	movl	-28(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, -24(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L21
.L3:
	movq	stderr(%rip), %rax
	movl	-20(%rbp), %edx
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$13, -16(%rbp)
	jmp	.L21
.L13:
	movl	-32(%rbp), %eax
	movl	-28(%rbp), %edi
	cltd
	idivl	%edi
	movl	%eax, -24(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L21
.L11:
	movl	$1, %eax
	jmp	.L34
.L6:
	movl	$0, %eax
	jmp	.L34
.L8:
	movl	$1, %eax
	jmp	.L34
.L16:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$16, %edx
	movl	$1, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$17, -16(%rbp)
	jmp	.L21
.L17:
	movl	$0, -32(%rbp)
	movl	$0, -28(%rbp)
	movl	$0, -24(%rbp)
	leaq	-28(%rbp), %rcx
	leaq	-32(%rbp), %rdx
	leaq	-33(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -20(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L21
.L14:
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -24(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L21
.L20:
	movl	$-1, %eax
	jmp	.L34
.L5:
	cmpl	$3, -20(%rbp)
	jne	.L31
	movq	$12, -16(%rbp)
	jmp	.L21
.L31:
	movq	$8, -16(%rbp)
	jmp	.L21
.L36:
	nop
.L21:
	jmp	.L33
.L34:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L35
	call	__stack_chk_fail@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	calculator, .-calculator
	.globl	main
	.type	main, @function
main:
.LFB5:
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
	movq	$0, _TIG_IZ_MtIt_envp(%rip)
	nop
.L38:
	movq	$0, _TIG_IZ_MtIt_argv(%rip)
	nop
.L39:
	movl	$0, _TIG_IZ_MtIt_argc(%rip)
	nop
	nop
.L40:
.L41:
#APP
# 111 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-MtIt--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_MtIt_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_MtIt_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_MtIt_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L51:
	cmpq	$4, -8(%rbp)
	je	.L42
	cmpq	$4, -8(%rbp)
	ja	.L53
	cmpq	$3, -8(%rbp)
	je	.L44
	cmpq	$3, -8(%rbp)
	ja	.L53
	cmpq	$0, -8(%rbp)
	je	.L45
	cmpq	$1, -8(%rbp)
	je	.L46
	jmp	.L53
.L42:
	cmpl	$0, -16(%rbp)
	je	.L47
	movq	$1, -8(%rbp)
	jmp	.L49
.L47:
	movq	$3, -8(%rbp)
	jmp	.L49
.L46:
	movl	$0, %eax
	jmp	.L52
.L44:
	call	calculator
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L49
.L45:
	movq	$3, -8(%rbp)
	jmp	.L49
.L53:
	nop
.L49:
	jmp	.L51
.L52:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
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
