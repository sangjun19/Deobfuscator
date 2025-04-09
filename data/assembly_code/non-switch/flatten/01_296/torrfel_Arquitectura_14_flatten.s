	.file	"torrfel_Arquitectura_14_flatten.c"
	.text
	.globl	_TIG_IZ_LXcT_argc
	.bss
	.align 4
	.type	_TIG_IZ_LXcT_argc, @object
	.size	_TIG_IZ_LXcT_argc, 4
_TIG_IZ_LXcT_argc:
	.zero	4
	.globl	_TIG_IZ_LXcT_argv
	.align 8
	.type	_TIG_IZ_LXcT_argv, @object
	.size	_TIG_IZ_LXcT_argv, 8
_TIG_IZ_LXcT_argv:
	.zero	8
	.globl	_TIG_IZ_LXcT_envp
	.align 8
	.type	_TIG_IZ_LXcT_envp, @object
	.size	_TIG_IZ_LXcT_envp, 8
_TIG_IZ_LXcT_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"PATH"
	.align 8
.LC1:
	.string	"No ha sido posible ejecutarlo. PATH: %s\n"
.LC2:
	.string	"Programa: "
.LC3:
	.string	" %s"
.LC4:
	.string	"salir"
.LC5:
	.string	"PID hijo: %d\n"
.LC6:
	.string	"Esperar? "
.LC7:
	.string	"si"
	.text
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
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_LXcT_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_LXcT_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_LXcT_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 96 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-LXcT--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_LXcT_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_LXcT_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_LXcT_envp(%rip)
	nop
	movq	$0, -88(%rbp)
.L39:
	cmpq	$23, -88(%rbp)
	ja	.L42
	movq	-88(%rbp), %rax
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
	.long	.L24-.L8
	.long	.L42-.L8
	.long	.L42-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L42-.L8
	.long	.L42-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L42-.L8
	.long	.L42-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L42-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L12:
	cmpl	$0, -108(%rbp)
	je	.L25
	movq	$5, -88(%rbp)
	jmp	.L27
.L25:
	movq	$19, -88(%rbp)
	jmp	.L27
.L22:
	movb	$1, -109(%rbp)
	movq	$11, -88(%rbp)
	jmp	.L27
.L16:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -88(%rbp)
	jmp	.L27
.L20:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-64(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -92(%rbp)
	movq	$22, -88(%rbp)
	jmp	.L27
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L40
	jmp	.L41
.L23:
	cmpl	$0, -100(%rbp)
	jne	.L29
	movq	$10, -88(%rbp)
	jmp	.L27
.L29:
	movq	$9, -88(%rbp)
	jmp	.L27
.L14:
	movl	$0, %edi
	call	exit@PLT
.L17:
	movzbl	-109(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L31
	movq	$8, -88(%rbp)
	jmp	.L27
.L31:
	movq	$23, -88(%rbp)
	jmp	.L27
.L19:
	movl	-108(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -88(%rbp)
	jmp	.L27
.L15:
	movl	-108(%rbp), %eax
	cmpl	-104(%rbp), %eax
	je	.L33
	movq	$10, -88(%rbp)
	jmp	.L27
.L33:
	movq	$11, -88(%rbp)
	jmp	.L27
.L11:
	leaq	-64(%rbp), %rcx
	leaq	-64(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	movl	%eax, -96(%rbp)
	movq	$20, -88(%rbp)
	jmp	.L27
.L13:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-69(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	fork@PLT
	movl	%eax, -108(%rbp)
	movq	$18, -88(%rbp)
	jmp	.L27
.L9:
	cmpl	$0, -92(%rbp)
	je	.L35
	movq	$17, -88(%rbp)
	jmp	.L27
.L35:
	movq	$4, -88(%rbp)
	jmp	.L27
.L21:
	leaq	-69(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -100(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L27
.L18:
	movl	$0, %edi
	call	wait@PLT
	movl	%eax, -104(%rbp)
	movq	$13, -88(%rbp)
	jmp	.L27
.L24:
	movb	$0, -109(%rbp)
	movq	$11, -88(%rbp)
	jmp	.L27
.L10:
	cmpl	$-1, -96(%rbp)
	jne	.L37
	movq	$12, -88(%rbp)
	jmp	.L27
.L37:
	movq	$16, -88(%rbp)
	jmp	.L27
.L42:
	nop
.L27:
	jmp	.L39
.L41:
	call	__stack_chk_fail@PLT
.L40:
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
