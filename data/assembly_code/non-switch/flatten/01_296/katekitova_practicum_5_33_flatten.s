	.file	"katekitova_practicum_5_33_flatten.c"
	.text
	.globl	_TIG_IZ_8qdk_argc
	.bss
	.align 4
	.type	_TIG_IZ_8qdk_argc, @object
	.size	_TIG_IZ_8qdk_argc, 4
_TIG_IZ_8qdk_argc:
	.zero	4
	.globl	_TIG_IZ_8qdk_argv
	.align 8
	.type	_TIG_IZ_8qdk_argv, @object
	.size	_TIG_IZ_8qdk_argv, 8
_TIG_IZ_8qdk_argv:
	.zero	8
	.globl	_TIG_IZ_8qdk_envp
	.align 8
	.type	_TIG_IZ_8qdk_envp, @object
	.size	_TIG_IZ_8qdk_envp, 8
_TIG_IZ_8qdk_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"'%c': %d\n"
.LC1:
	.string	"error "
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
	movq	$0, _TIG_IZ_8qdk_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8qdk_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8qdk_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 131 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8qdk--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_8qdk_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_8qdk_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_8qdk_envp(%rip)
	nop
	movq	$5, -48(%rbp)
.L18:
	cmpq	$5, -48(%rbp)
	ja	.L21
	movq	-48(%rbp), %rax
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
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	movb	$97, -18(%rbp)
	movb	$97, -17(%rbp)
	movb	$97, -16(%rbp)
	movb	$97, -15(%rbp)
	movb	$97, -14(%rbp)
	movb	$99, -13(%rbp)
	movb	$99, -12(%rbp)
	movb	$98, -11(%rbp)
	movb	$98, -10(%rbp)
	movb	$0, -9(%rbp)
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movl	%eax, %esi
	leaq	-52(%rbp), %rcx
	leaq	-53(%rbp), %rdx
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	f
	movq	$2, -48(%rbp)
	jmp	.L14
.L12:
	movl	-52(%rbp), %edx
	movzbl	-53(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -48(%rbp)
	jmp	.L14
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L7:
	movq	$4, -48(%rbp)
	jmp	.L14
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -48(%rbp)
	jmp	.L14
.L11:
	movzbl	-53(%rbp), %eax
	testb	%al, %al
	je	.L16
	movq	$1, -48(%rbp)
	jmp	.L14
.L16:
	movq	$0, -48(%rbp)
	jmp	.L14
.L21:
	nop
.L14:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	f
	.type	f, @function
f:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1104, %rsp
	movq	%rdi, -1080(%rbp)
	movl	%esi, -1084(%rbp)
	movq	%rdx, -1096(%rbp)
	movq	%rcx, -1104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$20, -1048(%rbp)
.L73:
	cmpq	$35, -1048(%rbp)
	ja	.L76
	movq	-1048(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L48-.L25
	.long	.L76-.L25
	.long	.L76-.L25
	.long	.L47-.L25
	.long	.L46-.L25
	.long	.L45-.L25
	.long	.L76-.L25
	.long	.L44-.L25
	.long	.L76-.L25
	.long	.L76-.L25
	.long	.L77-.L25
	.long	.L42-.L25
	.long	.L41-.L25
	.long	.L76-.L25
	.long	.L40-.L25
	.long	.L76-.L25
	.long	.L39-.L25
	.long	.L38-.L25
	.long	.L37-.L25
	.long	.L36-.L25
	.long	.L35-.L25
	.long	.L76-.L25
	.long	.L34-.L25
	.long	.L76-.L25
	.long	.L76-.L25
	.long	.L33-.L25
	.long	.L76-.L25
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L30-.L25
	.long	.L76-.L25
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L24-.L25
	.text
.L37:
	addl	$1, -1052(%rbp)
	movq	$33, -1048(%rbp)
	jmp	.L49
.L33:
	movl	-1056(%rbp), %eax
	movslq	%eax, %rdx
	movq	-1080(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$96, %al
	jle	.L50
	movq	$16, -1048(%rbp)
	jmp	.L49
.L50:
	movq	$34, -1048(%rbp)
	jmp	.L49
.L46:
	movq	-1096(%rbp), %rax
	movzbl	-1065(%rbp), %edx
	movb	%dl, (%rax)
	movq	-1104(%rbp), %rax
	movl	-1060(%rbp), %edx
	movl	%edx, (%rax)
	movq	$10, -1048(%rbp)
	jmp	.L49
.L40:
	movl	-1052(%rbp), %eax
	movb	%al, -1065(%rbp)
	movq	$18, -1048(%rbp)
	jmp	.L49
.L29:
	movl	-1084(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1060(%rbp)
	movb	$0, -1065(%rbp)
	movl	$0, -1056(%rbp)
	movq	$17, -1048(%rbp)
	jmp	.L49
.L41:
	movl	$0, -1040(%rbp)
	movl	$1, -1064(%rbp)
	movq	$22, -1048(%rbp)
	jmp	.L49
.L47:
	movl	-1052(%rbp), %eax
	cmpb	%al, -1065(%rbp)
	jle	.L52
	movq	$14, -1048(%rbp)
	jmp	.L49
.L52:
	movq	$18, -1048(%rbp)
	jmp	.L49
.L39:
	movl	-1056(%rbp), %eax
	movslq	%eax, %rdx
	movq	-1080(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$122, %al
	jg	.L54
	movq	$28, -1048(%rbp)
	jmp	.L49
.L54:
	movq	$34, -1048(%rbp)
	jmp	.L49
.L42:
	movl	-1056(%rbp), %eax
	movslq	%eax, %rdx
	movq	-1080(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$64, %al
	jle	.L56
	movq	$7, -1048(%rbp)
	jmp	.L49
.L56:
	movq	$25, -1048(%rbp)
	jmp	.L49
.L36:
	movl	-1064(%rbp), %eax
	movl	$0, -1040(%rbp,%rax,4)
	addl	$1, -1064(%rbp)
	movq	$22, -1048(%rbp)
	jmp	.L49
.L28:
	movl	$0, -1052(%rbp)
	movq	$33, -1048(%rbp)
	jmp	.L49
.L38:
	movl	-1056(%rbp), %eax
	cmpl	-1084(%rbp), %eax
	jge	.L58
	movq	$11, -1048(%rbp)
	jmp	.L49
.L58:
	movq	$32, -1048(%rbp)
	jmp	.L49
.L32:
	movl	-1056(%rbp), %eax
	movslq	%eax, %rdx
	movq	-1080(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movslq	%eax, %rdx
	movl	-1040(%rbp,%rdx,4), %edx
	addl	$1, %edx
	cltq
	movl	%edx, -1040(%rbp,%rax,4)
	movq	$34, -1048(%rbp)
	jmp	.L49
.L26:
	addl	$1, -1056(%rbp)
	movq	$17, -1048(%rbp)
	jmp	.L49
.L34:
	cmpl	$255, -1064(%rbp)
	jbe	.L60
	movq	$31, -1048(%rbp)
	jmp	.L49
.L60:
	movq	$19, -1048(%rbp)
	jmp	.L49
.L31:
	movl	-1056(%rbp), %eax
	movslq	%eax, %rdx
	movq	-1080(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movslq	%eax, %rdx
	movl	-1040(%rbp,%rdx,4), %edx
	addl	$1, %edx
	cltq
	movl	%edx, -1040(%rbp,%rax,4)
	movq	$34, -1048(%rbp)
	jmp	.L49
.L45:
	movl	-1052(%rbp), %eax
	cltq
	movl	-1040(%rbp,%rax,4), %eax
	cmpl	%eax, -1060(%rbp)
	jle	.L62
	movq	$0, -1048(%rbp)
	jmp	.L49
.L62:
	movq	$29, -1048(%rbp)
	jmp	.L49
.L27:
	cmpl	$255, -1052(%rbp)
	jg	.L64
	movq	$35, -1048(%rbp)
	jmp	.L49
.L64:
	movq	$4, -1048(%rbp)
	jmp	.L49
.L48:
	movl	-1052(%rbp), %eax
	cltq
	movl	-1040(%rbp,%rax,4), %eax
	movl	%eax, -1060(%rbp)
	movl	-1052(%rbp), %eax
	movb	%al, -1065(%rbp)
	movq	$18, -1048(%rbp)
	jmp	.L49
.L44:
	movl	-1056(%rbp), %eax
	movslq	%eax, %rdx
	movq	-1080(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$90, %al
	jg	.L67
	movq	$27, -1048(%rbp)
	jmp	.L49
.L67:
	movq	$25, -1048(%rbp)
	jmp	.L49
.L24:
	movl	-1052(%rbp), %eax
	cltq
	movl	-1040(%rbp,%rax,4), %eax
	testl	%eax, %eax
	jle	.L69
	movq	$5, -1048(%rbp)
	jmp	.L49
.L69:
	movq	$29, -1048(%rbp)
	jmp	.L49
.L30:
	movl	-1052(%rbp), %eax
	cltq
	movl	-1040(%rbp,%rax,4), %eax
	cmpl	%eax, -1060(%rbp)
	jne	.L71
	movq	$3, -1048(%rbp)
	jmp	.L49
.L71:
	movq	$18, -1048(%rbp)
	jmp	.L49
.L35:
	movq	$12, -1048(%rbp)
	jmp	.L49
.L76:
	nop
.L49:
	jmp	.L73
.L77:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L75
	call	__stack_chk_fail@PLT
.L75:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	f, .-f
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
