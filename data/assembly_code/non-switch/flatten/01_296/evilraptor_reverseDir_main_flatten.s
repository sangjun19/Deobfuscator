	.file	"evilraptor_reverseDir_main_flatten.c"
	.text
	.globl	_TIG_IZ_0zJA_envp
	.bss
	.align 8
	.type	_TIG_IZ_0zJA_envp, @object
	.size	_TIG_IZ_0zJA_envp, 8
_TIG_IZ_0zJA_envp:
	.zero	8
	.globl	_TIG_IZ_0zJA_argv
	.align 8
	.type	_TIG_IZ_0zJA_argv, @object
	.size	_TIG_IZ_0zJA_argv, 8
_TIG_IZ_0zJA_argv:
	.zero	8
	.globl	_TIG_IZ_0zJA_argc
	.align 4
	.type	_TIG_IZ_0zJA_argc, @object
	.size	_TIG_IZ_0zJA_argc, 4
_TIG_IZ_0zJA_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"malloc in get_full_name_target_file return:"
	.text
	.globl	get_full_name_target_file
	.type	get_full_name_target_file, @function
get_full_name_target_file:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	$8, -40(%rbp)
.L20:
	cmpq	$12, -40(%rbp)
	ja	.L21
	movq	-40(%rbp), %rax
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
	.long	.L21-.L4
	.long	.L9-.L4
	.long	.L21-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L21-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movl	%eax, -56(%rbp)
	movl	$0, -52(%rbp)
	movq	$5, -40(%rbp)
	jmp	.L14
.L7:
	movq	$0, -40(%rbp)
	jmp	.L14
.L12:
	movq	-48(%rbp), %rax
	jmp	.L15
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$2, -40(%rbp)
	jmp	.L14
.L5:
	movq	-72(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-80(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$1, -40(%rbp)
	jmp	.L14
.L9:
	movl	-56(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -52(%rbp)
	jge	.L16
	movq	$10, -40(%rbp)
	jmp	.L14
.L16:
	movq	$11, -40(%rbp)
	jmp	.L14
.L6:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -57(%rbp)
	movl	-56(%rbp), %eax
	subl	$1, %eax
	subl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movl	-52(%rbp), %edx
	movslq	%edx, %rcx
	movq	-80(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	movl	-56(%rbp), %eax
	subl	$1, %eax
	subl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-57(%rbp), %eax
	movb	%al, (%rdx)
	addl	$1, -52(%rbp)
	movq	$5, -40(%rbp)
	jmp	.L14
.L13:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$7, -40(%rbp)
	jmp	.L14
.L8:
	cmpq	$0, -48(%rbp)
	jne	.L18
	movq	$3, -40(%rbp)
	jmp	.L14
.L18:
	movq	$12, -40(%rbp)
	jmp	.L14
.L11:
	movl	$0, %eax
	jmp	.L15
.L21:
	nop
.L14:
	jmp	.L20
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	get_full_name_target_file, .-get_full_name_target_file
	.globl	get_original_dir_name
	.type	get_original_dir_name, @function
get_original_dir_name:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$14, -24(%rbp)
.L40:
	cmpq	$14, -24(%rbp)
	ja	.L41
	movq	-24(%rbp), %rax
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
	.long	.L41-.L25
	.long	.L33-.L25
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L41-.L25
	.long	.L41-.L25
	.long	.L30-.L25
	.long	.L41-.L25
	.long	.L41-.L25
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L41-.L25
	.long	.L24-.L25
	.text
.L24:
	movq	$12, -24(%rbp)
	jmp	.L34
.L26:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -40(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L34
.L33:
	movl	-36(%rbp), %eax
	cmpl	-40(%rbp), %eax
	jge	.L35
	movq	$11, -24(%rbp)
	jmp	.L34
.L35:
	movq	$10, -24(%rbp)
	jmp	.L34
.L31:
	movl	-40(%rbp), %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$47, %al
	jne	.L37
	movq	$9, -24(%rbp)
	jmp	.L34
.L37:
	movq	$6, -24(%rbp)
	jmp	.L34
.L27:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-36(%rbp), %edx
	movslq	%edx, %rcx
	movq	-32(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	addl	$1, -36(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L34
.L29:
	movq	-56(%rbp), %rax
	jmp	.L39
.L30:
	movl	-40(%rbp), %eax
	addl	$2, %eax
	cltq
	movl	$1, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	$0, -36(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L34
.L28:
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movb	$47, (%rax)
	movl	-40(%rbp), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$2, -24(%rbp)
	jmp	.L34
.L32:
	movq	-32(%rbp), %rax
	jmp	.L39
.L41:
	nop
.L34:
	jmp	.L40
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	get_original_dir_name, .-get_original_dir_name
	.section	.rodata
	.align 8
.LC1:
	.string	"calloc for tmp_dir in reverse_dir return:"
.LC2:
	.string	"done."
	.align 8
.LC3:
	.string	"opendir in reverse_dir return:"
	.text
	.globl	reverse_dir
	.type	reverse_dir, @function
reverse_dir:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	$23, -40(%rbp)
.L91:
	cmpq	$33, -40(%rbp)
	ja	.L92
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L70-.L45
	.long	.L69-.L45
	.long	.L68-.L45
	.long	.L92-.L45
	.long	.L67-.L45
	.long	.L92-.L45
	.long	.L66-.L45
	.long	.L65-.L45
	.long	.L92-.L45
	.long	.L64-.L45
	.long	.L92-.L45
	.long	.L92-.L45
	.long	.L63-.L45
	.long	.L62-.L45
	.long	.L61-.L45
	.long	.L60-.L45
	.long	.L59-.L45
	.long	.L58-.L45
	.long	.L57-.L45
	.long	.L56-.L45
	.long	.L92-.L45
	.long	.L93-.L45
	.long	.L92-.L45
	.long	.L54-.L45
	.long	.L53-.L45
	.long	.L52-.L45
	.long	.L92-.L45
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L49-.L45
	.long	.L93-.L45
	.long	.L93-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L57:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movw	$47, (%rax)
	movq	-64(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movw	$47, (%rax)
	movq	-80(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	create_dir
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movw	$47, (%rax)
	movq	-80(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	reverse_dir
	movq	$17, -40(%rbp)
	jmp	.L71
.L52:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	movq	$2, -40(%rbp)
	jmp	.L71
.L67:
	movq	-64(%rbp), %rax
	movzbl	18(%rax), %eax
	cmpb	$8, %al
	jne	.L72
	movq	$29, -40(%rbp)
	jmp	.L71
.L72:
	movq	$27, -40(%rbp)
	jmp	.L71
.L61:
	movl	$0, -92(%rbp)
	movq	$25, -40(%rbp)
	jmp	.L71
.L60:
	movw	$0, -94(%rbp)
	movq	-64(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	get_full_name_original_file
	movq	%rax, -80(%rbp)
	movq	-64(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-104(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	get_full_name_target_file
	movq	%rax, -88(%rbp)
	movq	$33, -40(%rbp)
	jmp	.L71
.L63:
	cmpq	$0, -56(%rbp)
	jne	.L75
	movq	$16, -40(%rbp)
	jmp	.L71
.L75:
	movq	$14, -40(%rbp)
	jmp	.L71
.L69:
	cmpw	$1, -94(%rbp)
	jne	.L77
	movq	$4, -40(%rbp)
	jmp	.L71
.L77:
	movq	$17, -40(%rbp)
	jmp	.L71
.L54:
	movq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -72(%rbp)
	movq	$24, -40(%rbp)
	jmp	.L71
.L59:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$21, -40(%rbp)
	jmp	.L71
.L53:
	cmpq	$0, -72(%rbp)
	jne	.L79
	movq	$6, -40(%rbp)
	jmp	.L71
.L79:
	movq	$0, -40(%rbp)
	jmp	.L71
.L64:
	cmpq	$0, -64(%rbp)
	je	.L81
	movq	$15, -40(%rbp)
	jmp	.L71
.L81:
	movq	$32, -40(%rbp)
	jmp	.L71
.L62:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-64(%rbp), %rax
	addq	$19, %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	addq	$2, %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$12, -40(%rbp)
	jmp	.L71
.L56:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$30, -40(%rbp)
	jmp	.L71
.L46:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$19, -40(%rbp)
	jmp	.L71
.L58:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -64(%rbp)
	movq	$9, -40(%rbp)
	jmp	.L71
.L66:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$31, -40(%rbp)
	jmp	.L71
.L51:
	movq	-64(%rbp), %rax
	movzbl	18(%rax), %eax
	cmpb	$4, %al
	jne	.L83
	movq	$13, -40(%rbp)
	jmp	.L71
.L83:
	movq	$17, -40(%rbp)
	jmp	.L71
.L50:
	movl	-92(%rbp), %eax
	movslq	%eax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	-92(%rbp), %edx
	movslq	%edx, %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	addl	$1, -92(%rbp)
	movq	$25, -40(%rbp)
	jmp	.L71
.L44:
	movq	-64(%rbp), %rax
	movzbl	19(%rax), %eax
	cmpb	$46, %al
	je	.L85
	movq	$7, -40(%rbp)
	jmp	.L71
.L85:
	movq	$1, -40(%rbp)
	jmp	.L71
.L70:
	cmpq	$0, -72(%rbp)
	je	.L87
	movq	$17, -40(%rbp)
	jmp	.L71
.L87:
	movq	$19, -40(%rbp)
	jmp	.L71
.L65:
	movw	$1, -94(%rbp)
	movq	$1, -40(%rbp)
	jmp	.L71
.L49:
	movq	-80(%rbp), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	create_reverse_file
	movq	$17, -40(%rbp)
	jmp	.L71
.L68:
	movl	-92(%rbp), %eax
	cltq
	movq	-48(%rbp), %rdx
	subq	$1, %rdx
	cmpq	%rdx, %rax
	jnb	.L89
	movq	$28, -40(%rbp)
	jmp	.L71
.L89:
	movq	$18, -40(%rbp)
	jmp	.L71
.L92:
	nop
.L71:
	jmp	.L91
.L93:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	reverse_dir, .-reverse_dir
	.section	.rodata
	.align 8
.LC4:
	.string	"you have to write directory name, error!"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, _TIG_IZ_0zJA_envp(%rip)
	nop
.L95:
	movq	$0, _TIG_IZ_0zJA_argv(%rip)
	nop
.L96:
	movl	$0, _TIG_IZ_0zJA_argc(%rip)
	nop
	nop
.L97:
.L98:
#APP
# 320 "evilraptor_reverseDir_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0zJA--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_0zJA_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_0zJA_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_0zJA_envp(%rip)
	nop
	movq	$4, -40(%rbp)
.L110:
	cmpq	$5, -40(%rbp)
	ja	.L111
	movq	-40(%rbp), %rax
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
	.long	.L111-.L101
	.long	.L105-.L101
	.long	.L104-.L101
	.long	.L103-.L101
	.long	.L102-.L101
	.long	.L100-.L101
	.text
.L102:
	cmpl	$1, -52(%rbp)
	jg	.L106
	movq	$2, -40(%rbp)
	jmp	.L108
.L106:
	movq	$1, -40(%rbp)
	jmp	.L108
.L105:
	movq	-64(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	get_original_dir_name
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	get_first_part
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	create_reverse_dir
	movq	-24(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	reverse_dir
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -40(%rbp)
	jmp	.L108
.L103:
	movl	$0, %eax
	jmp	.L109
.L100:
	movl	$0, %eax
	jmp	.L109
.L104:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -40(%rbp)
	jmp	.L108
.L111:
	nop
.L108:
	jmp	.L110
.L109:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC5:
	.string	"malloc in get_full_name_original_file return:"
	.text
	.globl	get_full_name_original_file
	.type	get_full_name_original_file, @function
get_full_name_original_file:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	$0, -32(%rbp)
.L126:
	cmpq	$7, -32(%rbp)
	ja	.L127
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L115(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L115(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L115:
	.long	.L121-.L115
	.long	.L120-.L115
	.long	.L119-.L115
	.long	.L118-.L115
	.long	.L117-.L115
	.long	.L127-.L115
	.long	.L116-.L115
	.long	.L114-.L115
	.text
.L117:
	cmpq	$0, -40(%rbp)
	jne	.L122
	movq	$1, -32(%rbp)
	jmp	.L124
.L122:
	movq	$7, -32(%rbp)
	jmp	.L124
.L120:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$2, -32(%rbp)
	jmp	.L124
.L118:
	movq	-40(%rbp), %rax
	jmp	.L125
.L116:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L124
.L121:
	movq	$6, -32(%rbp)
	jmp	.L124
.L114:
	movq	-56(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-64(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$3, -32(%rbp)
	jmp	.L124
.L119:
	movl	$0, %eax
	jmp	.L125
.L127:
	nop
.L124:
	jmp	.L126
.L125:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	get_full_name_original_file, .-get_full_name_original_file
	.section	.rodata
	.align 8
.LC6:
	.string	"fread in create_reverse_file return:"
	.align 8
.LC7:
	.string	"fclose(source) in create_reverse_file return:"
	.align 8
.LC8:
	.string	"fseek when going to the end of the file in create_reverse_file return:"
	.align 8
.LC9:
	.string	"fwrite in create_reverse_file return:"
	.align 8
.LC10:
	.string	"fwrite(target_dir) in create_reverse_file return:"
	.align 8
.LC11:
	.string	"cant set permissions,chmod in create_reverse_file return:"
	.align 8
.LC12:
	.string	"cant read source file,open in create_reverse_file return:"
.LC13:
	.string	"rb"
.LC14:
	.string	"wb"
	.align 8
.LC15:
	.string	"cant create file,open in create_reverse_file return:"
	.align 8
.LC16:
	.string	"close in create_reverse_file return:"
	.align 8
.LC17:
	.string	"cant get permissions,stat in create_reverse_file return:"
	.text
	.globl	create_reverse_file
	.type	create_reverse_file, @function
create_reverse_file:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$304, %rsp
	movq	%rdi, -296(%rbp)
	movq	%rsi, -304(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$53, -184(%rbp)
.L216:
	cmpq	$57, -184(%rbp)
	ja	.L219
	movq	-184(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L131(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L131(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L131:
	.long	.L185-.L131
	.long	.L184-.L131
	.long	.L183-.L131
	.long	.L182-.L131
	.long	.L181-.L131
	.long	.L180-.L131
	.long	.L179-.L131
	.long	.L178-.L131
	.long	.L219-.L131
	.long	.L177-.L131
	.long	.L176-.L131
	.long	.L220-.L131
	.long	.L220-.L131
	.long	.L173-.L131
	.long	.L220-.L131
	.long	.L219-.L131
	.long	.L171-.L131
	.long	.L170-.L131
	.long	.L169-.L131
	.long	.L168-.L131
	.long	.L220-.L131
	.long	.L220-.L131
	.long	.L165-.L131
	.long	.L164-.L131
	.long	.L163-.L131
	.long	.L220-.L131
	.long	.L161-.L131
	.long	.L220-.L131
	.long	.L220-.L131
	.long	.L220-.L131
	.long	.L157-.L131
	.long	.L156-.L131
	.long	.L155-.L131
	.long	.L220-.L131
	.long	.L153-.L131
	.long	.L152-.L131
	.long	.L151-.L131
	.long	.L150-.L131
	.long	.L149-.L131
	.long	.L148-.L131
	.long	.L147-.L131
	.long	.L220-.L131
	.long	.L145-.L131
	.long	.L144-.L131
	.long	.L143-.L131
	.long	.L220-.L131
	.long	.L141-.L131
	.long	.L140-.L131
	.long	.L139-.L131
	.long	.L138-.L131
	.long	.L137-.L131
	.long	.L136-.L131
	.long	.L135-.L131
	.long	.L134-.L131
	.long	.L219-.L131
	.long	.L133-.L131
	.long	.L220-.L131
	.long	.L130-.L131
	.text
.L169:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$45, -184(%rbp)
	jmp	.L186
.L137:
	movq	-240(%rbp), %rax
	movl	$2, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movl	%eax, -260(%rbp)
	movq	$39, -184(%rbp)
	jmp	.L186
.L138:
	movq	-296(%rbp), %rax
	movl	$511, %edx
	movl	$66, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -244(%rbp)
	movl	-244(%rbp), %eax
	movl	%eax, -276(%rbp)
	movq	$26, -184(%rbp)
	jmp	.L186
.L135:
	movq	-240(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	%eax, -252(%rbp)
	movq	$40, -184(%rbp)
	jmp	.L186
.L181:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$27, -184(%rbp)
	jmp	.L186
.L157:
	movl	-136(%rbp), %edx
	movq	-296(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	chmod@PLT
	movl	%eax, -264(%rbp)
	movq	$19, -184(%rbp)
	jmp	.L186
.L156:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$56, -184(%rbp)
	jmp	.L186
.L184:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$12, -184(%rbp)
	jmp	.L186
.L164:
	movq	-232(%rbp), %rdx
	leaq	-277(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	%rax, -192(%rbp)
	movq	$24, -184(%rbp)
	jmp	.L186
.L182:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$33, -184(%rbp)
	jmp	.L186
.L171:
	movq	-240(%rbp), %rdx
	leaq	-277(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -200(%rbp)
	movq	$34, -184(%rbp)
	jmp	.L186
.L163:
	cmpq	$0, -192(%rbp)
	jne	.L188
	movq	$7, -184(%rbp)
	jmp	.L186
.L188:
	movq	$52, -184(%rbp)
	jmp	.L186
.L151:
	cmpl	$-1, -256(%rbp)
	jne	.L190
	movq	$31, -184(%rbp)
	jmp	.L186
.L190:
	movq	$9, -184(%rbp)
	jmp	.L186
.L130:
	movq	-240(%rbp), %rax
	movl	$1, %edx
	movq	$-2, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movl	%eax, -256(%rbp)
	movq	$36, -184(%rbp)
	jmp	.L186
.L161:
	cmpl	$-1, -276(%rbp)
	jne	.L192
	movq	$0, -184(%rbp)
	jmp	.L186
.L192:
	movq	$37, -184(%rbp)
	jmp	.L186
.L177:
	movq	-240(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -208(%rbp)
	movq	$6, -184(%rbp)
	jmp	.L186
.L173:
	cmpl	$-1, -268(%rbp)
	jne	.L194
	movq	$43, -184(%rbp)
	jmp	.L186
.L194:
	movq	$30, -184(%rbp)
	jmp	.L186
.L136:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$29, -184(%rbp)
	jmp	.L186
.L168:
	cmpl	$-1, -264(%rbp)
	jne	.L196
	movq	$17, -184(%rbp)
	jmp	.L186
.L196:
	movq	$10, -184(%rbp)
	jmp	.L186
.L155:
	cmpl	$-1, -248(%rbp)
	jne	.L198
	movq	$51, -184(%rbp)
	jmp	.L186
.L198:
	movq	$28, -184(%rbp)
	jmp	.L186
.L170:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$20, -184(%rbp)
	jmp	.L186
.L147:
	cmpl	$-1, -252(%rbp)
	jne	.L200
	movq	$4, -184(%rbp)
	jmp	.L186
.L200:
	movq	$35, -184(%rbp)
	jmp	.L186
.L133:
	leaq	-160(%rbp), %rdx
	movq	-304(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -268(%rbp)
	movq	$13, -184(%rbp)
	jmp	.L186
.L179:
	cmpq	$0, -208(%rbp)
	jle	.L202
	movq	$42, -184(%rbp)
	jmp	.L186
.L202:
	movq	$16, -184(%rbp)
	jmp	.L186
.L149:
	cmpl	$-1, -272(%rbp)
	jne	.L204
	movq	$46, -184(%rbp)
	jmp	.L186
.L204:
	movq	$55, -184(%rbp)
	jmp	.L186
.L153:
	cmpq	$0, -200(%rbp)
	jne	.L206
	movq	$18, -184(%rbp)
	jmp	.L186
.L206:
	movq	$23, -184(%rbp)
	jmp	.L186
.L139:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$14, -184(%rbp)
	jmp	.L186
.L165:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$25, -184(%rbp)
	jmp	.L186
.L134:
	movq	$49, -184(%rbp)
	jmp	.L186
.L140:
	movq	-232(%rbp), %rdx
	leaq	-277(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	%rax, -216(%rbp)
	movq	$44, -184(%rbp)
	jmp	.L186
.L143:
	cmpq	$-1, -216(%rbp)
	jne	.L208
	movq	$3, -184(%rbp)
	jmp	.L186
.L208:
	movq	$57, -184(%rbp)
	jmp	.L186
.L180:
	cmpq	$0, -224(%rbp)
	jne	.L210
	movq	$1, -184(%rbp)
	jmp	.L186
.L210:
	movq	$47, -184(%rbp)
	jmp	.L186
.L150:
	movl	-276(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	%eax, -272(%rbp)
	movq	$38, -184(%rbp)
	jmp	.L186
.L176:
	movq	-304(%rbp), %rax
	leaq	.LC13(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -176(%rbp)
	movq	-176(%rbp), %rax
	movq	%rax, -240(%rbp)
	movq	-296(%rbp), %rax
	leaq	.LC14(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -168(%rbp)
	movq	-168(%rbp), %rax
	movq	%rax, -232(%rbp)
	movq	$2, -184(%rbp)
	jmp	.L186
.L145:
	movq	-240(%rbp), %rdx
	leaq	-277(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -224(%rbp)
	movq	$5, -184(%rbp)
	jmp	.L186
.L185:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$11, -184(%rbp)
	jmp	.L186
.L141:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$55, -184(%rbp)
	jmp	.L186
.L148:
	cmpl	$-1, -260(%rbp)
	jne	.L212
	movq	$22, -184(%rbp)
	jmp	.L186
.L212:
	movq	$9, -184(%rbp)
	jmp	.L186
.L178:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$21, -184(%rbp)
	jmp	.L186
.L152:
	movq	-232(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	%eax, -248(%rbp)
	movq	$32, -184(%rbp)
	jmp	.L186
.L144:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$41, -184(%rbp)
	jmp	.L186
.L183:
	cmpq	$0, -240(%rbp)
	jne	.L214
	movq	$48, -184(%rbp)
	jmp	.L186
.L214:
	movq	$50, -184(%rbp)
	jmp	.L186
.L219:
	nop
.L186:
	jmp	.L216
.L220:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L218
	call	__stack_chk_fail@PLT
.L218:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	create_reverse_file, .-create_reverse_file
	.section	.rodata
	.align 8
.LC18:
	.string	"cant set permissions, chmod in create_dir return:"
	.align 8
.LC19:
	.string	"cant get permissions, stat in create_dir return:"
.LC20:
	.string	"mkdir in create_dir return:"
	.text
	.globl	create_dir
	.type	create_dir, @function
create_dir:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movq	%rdi, -200(%rbp)
	movq	%rsi, -208(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$12, -168(%rbp)
.L245:
	cmpq	$12, -168(%rbp)
	ja	.L248
	movq	-168(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L224(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L224(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L224:
	.long	.L236-.L224
	.long	.L249-.L224
	.long	.L249-.L224
	.long	.L249-.L224
	.long	.L232-.L224
	.long	.L231-.L224
	.long	.L230-.L224
	.long	.L229-.L224
	.long	.L228-.L224
	.long	.L249-.L224
	.long	.L226-.L224
	.long	.L225-.L224
	.long	.L223-.L224
	.text
.L232:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$9, -168(%rbp)
	jmp	.L237
.L223:
	movq	-200(%rbp), %rax
	movl	$448, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movl	%eax, -180(%rbp)
	movq	$7, -168(%rbp)
	jmp	.L237
.L228:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -168(%rbp)
	jmp	.L237
.L225:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$1, -168(%rbp)
	jmp	.L237
.L230:
	cmpl	$-1, -176(%rbp)
	jne	.L239
	movq	$8, -168(%rbp)
	jmp	.L237
.L239:
	movq	$5, -168(%rbp)
	jmp	.L237
.L231:
	movl	-136(%rbp), %edx
	movq	-200(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	chmod@PLT
	movl	%eax, -172(%rbp)
	movq	$0, -168(%rbp)
	jmp	.L237
.L226:
	leaq	-160(%rbp), %rdx
	movq	-208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -176(%rbp)
	movq	$6, -168(%rbp)
	jmp	.L237
.L236:
	cmpl	$-1, -172(%rbp)
	jne	.L241
	movq	$4, -168(%rbp)
	jmp	.L237
.L241:
	movq	$2, -168(%rbp)
	jmp	.L237
.L229:
	cmpl	$-1, -180(%rbp)
	jne	.L243
	movq	$11, -168(%rbp)
	jmp	.L237
.L243:
	movq	$10, -168(%rbp)
	jmp	.L237
.L248:
	nop
.L237:
	jmp	.L245
.L249:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L247
	call	__stack_chk_fail@PLT
.L247:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	create_dir, .-create_dir
	.section	.rodata
	.align 8
.LC21:
	.string	"cant set permissions,chmod in create_reverse_dir return:"
	.align 8
.LC22:
	.string	"cant get permissions,stat in create_reverse_dir return:"
	.text
	.globl	create_reverse_dir
	.type	create_reverse_dir, @function
create_reverse_dir:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$192, %rsp
	movq	%rdi, -184(%rbp)
	movq	%rsi, -192(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -168(%rbp)
.L269:
	cmpq	$9, -168(%rbp)
	ja	.L272
	movq	-168(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L253(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L253(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L253:
	.long	.L262-.L253
	.long	.L261-.L253
	.long	.L273-.L253
	.long	.L273-.L253
	.long	.L258-.L253
	.long	.L257-.L253
	.long	.L256-.L253
	.long	.L255-.L253
	.long	.L273-.L253
	.long	.L252-.L253
	.text
.L258:
	movq	$7, -168(%rbp)
	jmp	.L263
.L261:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$8, -168(%rbp)
	jmp	.L263
.L252:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -168(%rbp)
	jmp	.L263
.L256:
	cmpl	$-1, -172(%rbp)
	jne	.L265
	movq	$1, -168(%rbp)
	jmp	.L263
.L265:
	movq	$2, -168(%rbp)
	jmp	.L263
.L257:
	cmpl	$-1, -176(%rbp)
	jne	.L267
	movq	$9, -168(%rbp)
	jmp	.L263
.L267:
	movq	$0, -168(%rbp)
	jmp	.L263
.L262:
	movl	-136(%rbp), %edx
	movq	-184(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	chmod@PLT
	movl	%eax, -172(%rbp)
	movq	$6, -168(%rbp)
	jmp	.L263
.L255:
	movq	-184(%rbp), %rax
	movl	$448, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	leaq	-160(%rbp), %rdx
	movq	-192(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -176(%rbp)
	movq	$5, -168(%rbp)
	jmp	.L263
.L272:
	nop
.L263:
	jmp	.L269
.L273:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L271
	call	__stack_chk_fail@PLT
.L271:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	create_reverse_dir, .-create_reverse_dir
	.section	.rodata
	.align 8
.LC23:
	.string	"calloc in get_first_part return:"
	.text
	.globl	get_first_part
	.type	get_first_part, @function
get_first_part:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	$25, -24(%rbp)
.L305:
	cmpq	$26, -24(%rbp)
	ja	.L306
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L277(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L277(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L277:
	.long	.L306-.L277
	.long	.L292-.L277
	.long	.L291-.L277
	.long	.L290-.L277
	.long	.L289-.L277
	.long	.L306-.L277
	.long	.L306-.L277
	.long	.L306-.L277
	.long	.L288-.L277
	.long	.L287-.L277
	.long	.L306-.L277
	.long	.L286-.L277
	.long	.L306-.L277
	.long	.L285-.L277
	.long	.L306-.L277
	.long	.L284-.L277
	.long	.L306-.L277
	.long	.L283-.L277
	.long	.L282-.L277
	.long	.L281-.L277
	.long	.L306-.L277
	.long	.L280-.L277
	.long	.L279-.L277
	.long	.L306-.L277
	.long	.L306-.L277
	.long	.L278-.L277
	.long	.L276-.L277
	.text
.L282:
	movl	-52(%rbp), %eax
	subl	$2, %eax
	subl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	-52(%rbp), %edx
	subl	$1, %edx
	movl	%edx, %ecx
	subl	-48(%rbp), %ecx
	movl	-36(%rbp), %edx
	addl	%ecx, %edx
	movslq	%edx, %rcx
	movq	-32(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	addl	$1, -36(%rbp)
	movq	$22, -24(%rbp)
	jmp	.L293
.L278:
	movq	$3, -24(%rbp)
	jmp	.L293
.L289:
	movl	-40(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jge	.L294
	movq	$17, -24(%rbp)
	jmp	.L293
.L294:
	movq	$13, -24(%rbp)
	jmp	.L293
.L284:
	movl	-52(%rbp), %eax
	subl	$2, %eax
	movl	%eax, -44(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L293
.L288:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$2, -24(%rbp)
	jmp	.L293
.L292:
	cmpq	$0, -32(%rbp)
	jne	.L296
	movq	$8, -24(%rbp)
	jmp	.L293
.L296:
	movq	$15, -24(%rbp)
	jmp	.L293
.L290:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -52(%rbp)
	movl	$0, -48(%rbp)
	movl	-52(%rbp), %eax
	addl	$1, %eax
	cltq
	movl	$1, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L293
.L280:
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$47, %al
	jne	.L298
	movq	$9, -24(%rbp)
	jmp	.L293
.L298:
	movq	$26, -24(%rbp)
	jmp	.L293
.L276:
	addl	$1, -48(%rbp)
	subl	$1, -44(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L293
.L286:
	movq	-32(%rbp), %rax
	jmp	.L300
.L287:
	movl	$0, -40(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L293
.L285:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$0, -36(%rbp)
	movq	$22, -24(%rbp)
	jmp	.L293
.L281:
	cmpl	$0, -44(%rbp)
	js	.L301
	movq	$21, -24(%rbp)
	jmp	.L293
.L301:
	movq	$9, -24(%rbp)
	jmp	.L293
.L283:
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	-40(%rbp), %edx
	movslq	%edx, %rcx
	movq	-32(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	addl	$1, -40(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L293
.L279:
	movl	-36(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jge	.L303
	movq	$18, -24(%rbp)
	jmp	.L293
.L303:
	movq	$11, -24(%rbp)
	jmp	.L293
.L291:
	movl	$0, %eax
	jmp	.L300
.L306:
	nop
.L293:
	jmp	.L305
.L300:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	get_first_part, .-get_first_part
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
