	.file	"Matthew-LA_UML_main_flatten.c"
	.text
	.globl	_TIG_IZ_EAGT_argc
	.bss
	.align 4
	.type	_TIG_IZ_EAGT_argc, @object
	.size	_TIG_IZ_EAGT_argc, 4
_TIG_IZ_EAGT_argc:
	.zero	4
	.globl	_TIG_IZ_EAGT_argv
	.align 8
	.type	_TIG_IZ_EAGT_argv, @object
	.size	_TIG_IZ_EAGT_argv, 8
_TIG_IZ_EAGT_argv:
	.zero	8
	.globl	_TIG_IZ_EAGT_envp
	.align 8
	.type	_TIG_IZ_EAGT_envp, @object
	.size	_TIG_IZ_EAGT_envp, 8
_TIG_IZ_EAGT_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%c"
.LC1:
	.string	"Please enter a string:"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_EAGT_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_EAGT_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_EAGT_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EAGT--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_EAGT_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_EAGT_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_EAGT_envp(%rip)
	nop
	movq	$5, -104(%rbp)
.L25:
	cmpq	$14, -104(%rbp)
	ja	.L28
	movq	-104(%rbp), %rax
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
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	movzbl	-109(%rbp), %eax
	cmpb	$10, %al
	jne	.L17
	movq	$3, -104(%rbp)
	jmp	.L19
.L17:
	movq	$13, -104(%rbp)
	jmp	.L19
.L12:
	leaq	-109(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -104(%rbp)
	jmp	.L19
.L15:
	movl	-108(%rbp), %eax
	cltq
	movb	$0, -96(%rbp,%rax)
	leaq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	reverseString
	leaq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -104(%rbp)
	jmp	.L19
.L11:
	cmpl	$79, -108(%rbp)
	jg	.L20
	movq	$8, -104(%rbp)
	jmp	.L19
.L20:
	movq	$3, -104(%rbp)
	jmp	.L19
.L9:
	movzbl	-109(%rbp), %edx
	movl	-108(%rbp), %eax
	cltq
	movb	%dl, -96(%rbp,%rax)
	addl	$1, -108(%rbp)
	movq	$7, -104(%rbp)
	jmp	.L19
.L14:
	movq	$0, -104(%rbp)
	jmp	.L19
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L16:
	movl	$0, -108(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -104(%rbp)
	jmp	.L19
.L13:
	movzbl	-109(%rbp), %eax
	cmpb	$10, %al
	je	.L23
	movq	$9, -104(%rbp)
	jmp	.L19
.L23:
	movq	$3, -104(%rbp)
	jmp	.L19
.L28:
	nop
.L19:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	reverseString
	.type	reverseString, @function
reverseString:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -56(%rbp)
	movq	$12, -8(%rbp)
.L50:
	cmpq	$16, -8(%rbp)
	ja	.L52
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L41-.L32
	.long	.L52-.L32
	.long	.L52-.L32
	.long	.L40-.L32
	.long	.L52-.L32
	.long	.L52-.L32
	.long	.L39-.L32
	.long	.L38-.L32
	.long	.L37-.L32
	.long	.L52-.L32
	.long	.L36-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L52-.L32
	.long	.L33-.L32
	.long	.L52-.L32
	.long	.L31-.L32
	.text
.L33:
	addl	$1, -32(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L42
.L34:
	movl	$0, -32(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L42
.L37:
	movl	$0, -32(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L42
.L40:
	movl	-36(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -32(%rbp)
	jge	.L43
	movq	$0, -8(%rbp)
	jmp	.L42
.L43:
	movq	$8, -8(%rbp)
	jmp	.L42
.L31:
	movl	-36(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -32(%rbp)
	jge	.L45
	movq	$7, -8(%rbp)
	jmp	.L42
.L45:
	movq	$10, -8(%rbp)
	jmp	.L42
.L35:
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L47
	movq	$14, -8(%rbp)
	jmp	.L42
.L47:
	movq	$6, -8(%rbp)
	jmp	.L42
.L39:
	movl	-32(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -16(%rbp)
	movl	$0, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L42
.L36:
	movq	-56(%rbp), %rax
	jmp	.L51
.L41:
	addq	$1, -16(%rbp)
	addl	$1, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L42
.L38:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, -28(%rbp)
	movq	-24(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-16(%rbp), %rax
	movb	%dl, (%rax)
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	movq	-24(%rbp), %rax
	movb	%dl, (%rax)
	addq	$1, -24(%rbp)
	subq	$1, -16(%rbp)
	addl	$1, -32(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L42
.L52:
	nop
.L42:
	jmp	.L50
.L51:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	reverseString, .-reverseString
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
