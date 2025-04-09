	.file	"chandramaniadil_Codevita-2015_A_flatten.c"
	.text
	.globl	_TIG_IZ_2OTv_argv
	.bss
	.align 8
	.type	_TIG_IZ_2OTv_argv, @object
	.size	_TIG_IZ_2OTv_argv, 8
_TIG_IZ_2OTv_argv:
	.zero	8
	.globl	_TIG_IZ_2OTv_argc
	.align 4
	.type	_TIG_IZ_2OTv_argc, @object
	.size	_TIG_IZ_2OTv_argc, 4
_TIG_IZ_2OTv_argc:
	.zero	4
	.globl	_TIG_IZ_2OTv_envp
	.align 8
	.type	_TIG_IZ_2OTv_envp, @object
	.size	_TIG_IZ_2OTv_envp, 8
_TIG_IZ_2OTv_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"Thank God"
.LC2:
	.string	"%d F\n"
.LC3:
	.string	"%d B\n"
.LC4:
	.string	"%d %d %d %d %d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_2OTv_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_2OTv_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_2OTv_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 136 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-2OTv--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_2OTv_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_2OTv_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_2OTv_envp(%rip)
	nop
	movq	$14, -16(%rbp)
.L57:
	cmpq	$38, -16(%rbp)
	ja	.L60
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L60-.L8
	.long	.L32-.L8
	.long	.L60-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L60-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L60-.L8
	.long	.L13-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L12-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L60-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movl	$0, -24(%rbp)
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L33
.L30:
	movl	-40(%rbp), %edx
	movl	-52(%rbp), %eax
	cmpl	%eax, %edx
	jg	.L34
	movq	$34, -16(%rbp)
	jmp	.L33
.L34:
	movq	$20, -16(%rbp)
	jmp	.L33
.L23:
	movq	$25, -16(%rbp)
	jmp	.L33
.L22:
	cmpl	$0, -28(%rbp)
	je	.L36
	movq	$35, -16(%rbp)
	jmp	.L33
.L36:
	movq	$26, -16(%rbp)
	jmp	.L33
.L12:
	movl	$0, -24(%rbp)
	movl	$1, -32(%rbp)
	movl	$1, -28(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L33
.L25:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L33
.L27:
	movl	-36(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-36(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -36(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L33
.L32:
	movl	-44(%rbp), %edx
	movl	-40(%rbp), %eax
	addl	%eax, %edx
	movl	-56(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L38
	movq	$12, -16(%rbp)
	jmp	.L33
.L38:
	movq	$28, -16(%rbp)
	jmp	.L33
.L17:
	cmpl	$0, -20(%rbp)
	jle	.L40
	movq	$37, -16(%rbp)
	jmp	.L33
.L40:
	movq	$13, -16(%rbp)
	jmp	.L33
.L31:
	movl	-52(%rbp), %edx
	movl	-44(%rbp), %eax
	addl	%eax, %edx
	movl	-56(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L42
	movq	$22, -16(%rbp)
	jmp	.L33
.L42:
	movq	$4, -16(%rbp)
	jmp	.L33
.L16:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L33
.L19:
	movl	-40(%rbp), %edx
	movl	-52(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L44
	movq	$1, -16(%rbp)
	jmp	.L33
.L44:
	movq	$28, -16(%rbp)
	jmp	.L33
.L14:
	cmpl	$0, -32(%rbp)
	je	.L46
	movq	$9, -16(%rbp)
	jmp	.L33
.L46:
	movq	$24, -16(%rbp)
	jmp	.L33
.L26:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L33
.L24:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L58
	jmp	.L59
.L21:
	movl	-48(%rbp), %edx
	movl	-40(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L33
.L28:
	movl	-56(%rbp), %edx
	movl	-52(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L49
	movq	$21, -16(%rbp)
	jmp	.L33
.L49:
	movq	$31, -16(%rbp)
	jmp	.L33
.L7:
	movl	-48(%rbp), %edx
	movl	-44(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L33
.L11:
	movl	-48(%rbp), %edx
	movl	-40(%rbp), %eax
	imull	%edx, %eax
	addl	%eax, -24(%rbp)
	movl	$0, -28(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L33
.L18:
	movl	-40(%rbp), %eax
	movl	-52(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, %ecx
	movl	-56(%rbp), %eax
	addl	%ecx, %eax
	movl	%eax, -40(%rbp)
	movl	-44(%rbp), %edx
	movl	-52(%rbp), %eax
	addl	%edx, %eax
	movl	-56(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, -44(%rbp)
	movl	-56(%rbp), %edx
	movl	-52(%rbp), %eax
	addl	%eax, %edx
	movl	-48(%rbp), %eax
	imull	%edx, %eax
	addl	%eax, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L33
.L13:
	movl	-40(%rbp), %edx
	movl	-52(%rbp), %eax
	cmpl	%eax, %edx
	jg	.L51
	movq	$19, -16(%rbp)
	jmp	.L33
.L51:
	movq	$38, -16(%rbp)
	jmp	.L33
.L29:
	cmpl	$0, -32(%rbp)
	je	.L53
	movq	$15, -16(%rbp)
	jmp	.L33
.L53:
	movq	$26, -16(%rbp)
	jmp	.L33
.L9:
	leaq	-40(%rbp), %rdi
	leaq	-44(%rbp), %rsi
	leaq	-48(%rbp), %rcx
	leaq	-52(%rbp), %rdx
	leaq	-56(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L10:
	movl	-40(%rbp), %edx
	movl	-52(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L55
	movq	$3, -16(%rbp)
	jmp	.L33
.L55:
	movq	$4, -16(%rbp)
	jmp	.L33
.L20:
	movl	-40(%rbp), %eax
	movl	-52(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, -40(%rbp)
	movl	-44(%rbp), %edx
	movl	-52(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -44(%rbp)
	movl	-52(%rbp), %edx
	movl	-44(%rbp), %eax
	addl	%eax, %edx
	movl	-48(%rbp), %eax
	imull	%edx, %eax
	addl	%eax, -24(%rbp)
	movl	$0, -32(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L33
.L60:
	nop
.L33:
	jmp	.L57
.L59:
	call	__stack_chk_fail@PLT
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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
