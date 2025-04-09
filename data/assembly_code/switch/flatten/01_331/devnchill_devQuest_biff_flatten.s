	.file	"devnchill_devQuest_biff_flatten.c"
	.text
	.globl	_TIG_IZ_T88X_envp
	.bss
	.align 8
	.type	_TIG_IZ_T88X_envp, @object
	.size	_TIG_IZ_T88X_envp, 8
_TIG_IZ_T88X_envp:
	.zero	8
	.globl	_TIG_IZ_T88X_argc
	.align 4
	.type	_TIG_IZ_T88X_argc, @object
	.size	_TIG_IZ_T88X_argc, 4
_TIG_IZ_T88X_argc:
	.zero	4
	.globl	_TIG_IZ_T88X_argv
	.align 8
	.type	_TIG_IZ_T88X_argv, @object
	.size	_TIG_IZ_T88X_argv, 8
_TIG_IZ_T88X_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter message: "
.LC1:
	.string	" !!!!!!!!!!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_T88X_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_T88X_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_T88X_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 130 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-T88X--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_T88X_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_T88X_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_T88X_envp(%rip)
	nop
	movq	$13, -72(%rbp)
.L45:
	cmpq	$34, -72(%rbp)
	ja	.L48
	movq	-72(%rbp), %rax
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
	.long	.L27-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L48-.L8
	.long	.L22-.L8
	.long	.L48-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L48-.L8
	.long	.L48-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L48-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L48-.L8
	.long	.L11-.L8
	.long	.L48-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L48-.L8
	.long	.L7-.L8
	.text
.L18:
	movl	$0, -92(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -88(%rbp)
	movq	$31, -72(%rbp)
	jmp	.L28
.L14:
	movb	$48, -93(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L28
.L20:
	movb	$51, -93(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L28
.L19:
	movb	$52, -93(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L28
.L10:
	cmpl	$49, -88(%rbp)
	jg	.L29
	movq	$9, -72(%rbp)
	jmp	.L28
.L29:
	movq	$32, -72(%rbp)
	jmp	.L28
.L24:
	movb	$49, -93(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L28
.L15:
	movb	$53, -93(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L28
.L13:
	movq	$11, -72(%rbp)
	jmp	.L28
.L22:
	movl	-88(%rbp), %eax
	cltq
	movzbl	-93(%rbp), %edx
	movb	%dl, -64(%rbp,%rax)
	addl	$1, -92(%rbp)
	addl	$1, -88(%rbp)
	movq	$31, -72(%rbp)
	jmp	.L28
.L23:
	call	getchar@PLT
	movl	%eax, -76(%rbp)
	movl	-76(%rbp), %eax
	movb	%al, -93(%rbp)
	movq	$19, -72(%rbp)
	jmp	.L28
.L21:
	movq	$18, -72(%rbp)
	jmp	.L28
.L17:
	cmpb	$10, -93(%rbp)
	jne	.L31
	movq	$32, -72(%rbp)
	jmp	.L28
.L31:
	movq	$7, -72(%rbp)
	jmp	.L28
.L9:
	movl	$0, -80(%rbp)
	movq	$34, -72(%rbp)
	jmp	.L28
.L26:
	movl	-84(%rbp), %eax
	subl	$41, %eax
	cmpl	$42, %eax
	ja	.L33
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L35(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L35(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L35:
	.long	.L40-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L39-.L35
	.long	.L38-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L37-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L36-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L33-.L35
	.long	.L34-.L35
	.text
.L34:
	movq	$23, -72(%rbp)
	jmp	.L41
.L40:
	movq	$25, -72(%rbp)
	jmp	.L41
.L36:
	movq	$8, -72(%rbp)
	jmp	.L41
.L37:
	movq	$14, -72(%rbp)
	jmp	.L41
.L38:
	movq	$29, -72(%rbp)
	jmp	.L41
.L39:
	movq	$15, -72(%rbp)
	jmp	.L41
.L33:
	movq	$26, -72(%rbp)
	nop
.L41:
	jmp	.L28
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L46
	jmp	.L47
.L7:
	movl	-80(%rbp), %eax
	cmpl	-92(%rbp), %eax
	jge	.L43
	movq	$0, -72(%rbp)
	jmp	.L28
.L43:
	movq	$22, -72(%rbp)
	jmp	.L28
.L16:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$27, -72(%rbp)
	jmp	.L28
.L27:
	movl	-80(%rbp), %eax
	cltq
	movzbl	-64(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -80(%rbp)
	movq	$34, -72(%rbp)
	jmp	.L28
.L25:
	movsbl	-93(%rbp), %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -84(%rbp)
	movq	$6, -72(%rbp)
	jmp	.L28
.L11:
	movb	$56, -93(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L28
.L48:
	nop
.L28:
	jmp	.L45
.L47:
	call	__stack_chk_fail@PLT
.L46:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
